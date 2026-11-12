"""
Compare mkwellipf with the MKWELLIPF function of iMOD5.

Meant to be run locally, as it requires the iMOD5 executable.

Starting iMOD5 is slow, so a case is one IPF with many wells, each named after the
rule it probes. Settings that apply to all wells (MINKHT, MINKD, ISS, ...) differ
between cases.

Both implementations run the same ini file, in ``pytest_tmp_path/<case>/imod5/``
and ``pytest_tmp_path/<case>/ours/``. All ``imod_mkipf_*.ipf`` files are compared
(names, columns, rows, values). The case folder keeps the inputs, results,
``comparison.txt`` and ``diff.txt``.

Cases with a known difference from iMOD5 are listed in ``KNOWN_DIFFERENCES`` and
reported as xfail when they differ.

Environment variables:

* ``IMOD5_EXE``: path to the iMOD5 executable.

The tests are skipped when the executable does not exist.
"""

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import imod
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.mkwellipf import mkwellipf
from src.settings import read_mkwellipf_settings
from tests.imod5_utils import IMOD5_EXE, IMOD5_MARKS, run_imod5

pytestmark = IMOD5_MARKS

NODATA = -99999.0
RTOL, ATOL = 1e-4, 1e-3
# Layer values written next to each well. iMOD5 writes 1e20 in them for a well
# outside the grid, where ours writes the FNODATA value.
LAYER_COLUMNS = {"TOP", "BOT", "KD"}
IMOD5_NO_VALUE = 1e19
SHOWN = 20  # differences shown per column


class Well(NamedTuple):
    id: str
    top: float  # of the screen
    bottom: float
    q: float = -100.0
    x: float = 15.0
    y: float = 15.0


# A .txt time series: (yyyymmdd, rate) records, nodata is -999.99
Records = list[tuple[int, float]]


@dataclass(frozen=True)
class Model:
    """
    Touching layers on a grid with its origin at (0, 0). Arrays have shape
    (nrows, ncols), row 0 is the north. `k` is the horizontal conductivity (KHVIDF)
    of each layer.
    """

    top: list[np.ndarray]
    bottom: list[np.ndarray]
    k: list[np.ndarray] | None = None
    cellsize: float = 10.0


@dataclass(frozen=True)
class Case:
    wells: list[Well]
    model: Model | None  # None: no layers, only the rate of each well is determined
    ini: dict[str, str] = field(default_factory=dict)  # extra keywords
    series: dict[str, Records] | None = None  # time series by well id
    xfail: str = ""  # reason, if iMOD5 5.6.1 is known to give a different result


def flat_model(k: list[float] | None) -> Model:
    """
    Three layers on a 4 x 4 grid of 10 m cells, as in test_mkwellipf, with
    (top, bottom, K): (0, -10, 10), (-10, -20, 2), (-20, -40, 25).
    """

    def layers(values: list[float]) -> list[np.ndarray]:
        return [np.full((4, 4), value, dtype=np.float32) for value in values]

    conductivity = None if k is None else layers(k)
    return Model(layers([0, -10, -20]), layers([-10, -20, -40]), conductivity)


def varied_model(nlayer: int = 5, size: int = 20, seed: int = 42) -> Model:
    """
    Five layers on a 20 x 20 grid of 25 m cells below an undulating surface (5 to
    15 m): 1 to 15 m thick, 10% of them pinched out to 0, with log-normal
    conductivities.
    """
    rng = np.random.default_rng(seed)
    cellsize = 25.0
    x = (np.arange(size) + 0.5) * cellsize
    y = (size - 0.5 - np.arange(size)) * cellsize
    xx, yy = np.meshgrid(x, y)
    surface = 10.0 + 3.0 * np.sin(xx / 100.0) + 2.0 * np.cos(yy / 80.0)
    thickness = rng.uniform(1.0, 15.0, (nlayer, size, size))
    thickness[rng.random((nlayer, size, size)) < 0.1] = 0.0
    bottom = surface - np.cumsum(thickness, axis=0)
    top = np.concatenate([surface[None], bottom[:-1]])
    k = np.exp(rng.normal(0.0, 1.5, (nlayer, size, size)))
    return Model(
        top=list(top.astype(np.float32)),
        bottom=list(bottom.astype(np.float32)),
        k=list(k.astype(np.float32)),
        cellsize=cellsize,
    )


def varied_wells(n: int, seed: int = 7) -> list[Well]:
    """
    Wells at random places, some outside the 500 x 500 m grid, with random screens
    (some points) that are never entirely above the model, and 10% without a screen.
    """
    rng = np.random.default_rng(seed)
    wells = []
    for i in range(n):
        x, y = rng.uniform(-25.0, 525.0, 2)
        q = round(rng.uniform(-500, -1), 2)
        centre, length = rng.uniform(-60.0, 0.0), rng.uniform(0.0, 30.0)
        top, bottom = round(centre + length / 2, 2), round(centre - length / 2, 2)
        if rng.random() < 0.1:
            top = bottom = NODATA
        wells.append(Well(f"w{i}", top, bottom, q, x, y))
    return wells


def screen_probes(nodata: float = NODATA) -> list[Well]:
    """
    One well per rule, all in the same cell of flat_model. Its layers are
    1: 0 to -10 (K 10), 2: -10 to -20 (K 2) and 3: -20 to -40 (K 25) m.
    """
    return [
        # How the rate is split over the layers that the screen overlaps
        Well("split-over-layers", -5, -35),  # layer 1 to 3, weighed by length and K
        Well("one-layer", -2, -8),  # inside layer 1
        Well("is-layer", -10, -20),  # exactly layer 2
        Well("from-layer-boundary", -10, -30),  # layer 2 and 3, starting on a boundary
        Well("sliver-in-layer", -5, -10.000001),  # 1 micrometre into layer 2: rate 0
        # Screens that reach beyond the model
        Well("covers-model", 5, -60),  # above and below the model
        Well("top-above-model", 5, -15),  # starts above the model
        Well("bottom-below-model", -30, -60),  # ends below the model
        Well("below-model", -70, -80),  # entirely below: unassigned
        # Screens of zero length
        Well("point", -15, -15),  # in layer 2
        Well("point-on-boundary", -10, -10),  # between layer 1 and 2
        # Missing screen depths
        Well("unscreened", nodata, nodata),  # unassigned, or the first layer with MINKD
        Well("top-unknown", -5, nodata),  # one depth unknown counts as unscreened
        # Other properties of the well
        Well("outside-grid", -5, -35, x=100.0, y=100.0),  # unassigned, no layer values
        Well("injection", -5, -35, q=50.0),  # positive rate
        Well("zero-rate", -5, -35, q=0.0),
    ]


# Time series with one record per line (yyyymmdd, rate), one well per series. With ISS=0
# the rate is the average of all records, with ISS=1 the time average over 1 to 11
# January 2020. See test_time_series_average in test_mkwellipf.
SERIES: dict[str, Records] = {
    "plain": [(20200102, -10), (20200201, -30), (20200301, -20)],
    "all-nodata": [(20200102, -999.99), (20200201, -999.99)],  # average 0
    "empty": [],  # average 0
    "record-before-window": [(20191201, -10), (20200104, -20), (20200109, 0)],
    "first-record-in-window": [(20200106, -10)],  # 0 before the first record
    "inner-nodata": [(20200101, -10), (20200104, -999.99), (20200108, -30)],
    "leading-nodata": [(20200104, -999.99), (20200108, -20)],
    "trailing-nodata": [(20200101, -10), (20200106, -999.99)],
    "after-window": [(20200301, -10)],  # starts after the window
}
SERIES_WELLS = [Well(name, -5, -35) for name in SERIES]
SHORT_WINDOW = {"ISS": "1", "SDATE": "20200101", "EDATE": "20200111"}
VARIED_WELLS = varied_wells(300)

KHV = flat_model([10.0, 2.0, 25.0])

CASES: dict[str, Case] = {
    "default": Case(screen_probes(), KHV),
    # MINKHT: layers with K * thickness below 30 get no rate. MINKD is ignored without
    # KDVIDF. Point screens differ with MINKHT, see known-differences.
    "minkht": Case(
        [w for w in screen_probes() if not w.id.startswith("point")],
        KHV,
        {"MINKHT": "30", "MINKD": "40"},
    ),
    # Without K the rate is split by screen length, MINKHT and MINKD are ignored
    "no-k": Case(screen_probes(), flat_model(None), {"MINKHT": "30", "MINKD": "40"}),
    # Screen depths with nodata (FNODATA) and layers without a rate (HNODATA)
    "nodata-values": Case(
        screen_probes(-999.0), KHV, {"FNODATA": "-999", "HNODATA": "-9999"}
    ),
    "series-iss0": Case(SERIES_WELLS, KHV, series=SERIES),
    "series-iss1": Case(SERIES_WELLS, KHV, SHORT_WINDOW, SERIES),
    "series-iss1-no-layers": Case(SERIES_WELLS, None, SHORT_WINDOW, SERIES),
    # 300 random wells on layers with pinch-outs
    "varied": Case(VARIED_WELLS, varied_model()),
    "known-differences": Case(
        [
            Well("above-model", 5, 2),  # iMOD5: all to layer 1
            Well("reversed", -35, -5),  # top below bottom, iMOD5: all to layer 1
            Well("point", -15, -15),  # iMOD5: layer 2, also if MINKHT excludes it
            Well("point-on-boundary", -10, -10),  # same, layer 2
        ],
        KHV,
        {"MINKHT": "30"},
    ),
}
# MINKD with KDVIDF is not compared: iMOD5 5.6.1 takes a constant KDVIDF as
# thickness / kd instead of kd / thickness, and crashes on KDVIDF IDF files. It is
# only covered by test_mkwellipf.py.

# Cases where our result is known to differ from iMOD5 5.6.1: reported as xfail.
KNOWN_DIFFERENCES = {
    "known-differences": (
        "iMOD5 gives these wells to a layer, we leave them unassigned: a screen "
        "above the model or reversed (to layer 1), and with MINKHT a point screen "
        "(to the layer at the top of the screen)."
    ),
}


def write_layers(
    folder: Path, name: str, arrays: list[np.ndarray], cellsize: float
) -> list[Path]:
    """Write one IDF per layer, ``<name>_l1.idf``, ``<name>_l2.idf``, ..."""
    nrows, ncols = arrays[0].shape
    coords = {
        "layer": range(1, len(arrays) + 1),
        "y": (nrows - 0.5 - np.arange(nrows)) * cellsize,
        "x": (np.arange(ncols) + 0.5) * cellsize,
        "dx": cellsize,
        "dy": -cellsize,
    }
    data = xr.DataArray(np.stack(arrays), coords, ("layer", "y", "x"))
    imod.idf.save(folder / name, data)
    return [folder / f"{name}_l{i}.idf" for i in range(1, len(arrays) + 1)]


def write_model(folder: Path, model: Model) -> list[str]:
    """Write the IDF files of the layers and return the ini lines that refer to them."""
    folder.mkdir(parents=True, exist_ok=True)
    layers = {
        "TOPIDF": write_layers(folder, "top", model.top, model.cellsize),
        "BOTIDF": write_layers(folder, "bot", model.bottom, model.cellsize),
    }
    if model.k is not None:
        layers["KHVIDF"] = write_layers(folder, "k", model.k, model.cellsize)
    lines = [f"NLAY={len(model.top)}"]
    for keyword, files in layers.items():
        lines += [f"{keyword}{i}={path}" for i, path in enumerate(files, start=1)]
    return lines


def write_inputs(folder: Path, case: Case, model_lines: list[str]) -> Path:
    """Write the IPF, the time series and the ini file. Returns the ini file."""
    folder.mkdir(parents=True, exist_ok=True)
    ipf = folder / "wells.ipf"
    columns = ["X", "Y", "ID", "Z_TOP", "Z_BOT", "Q"]
    # With time series, column 3 (ID) holds the names of the .txt files
    association = f"{3 if case.series else 0},txt"
    header = [str(len(case.wells)), str(len(columns)), *columns, association]
    rows = [f"{w.x},{w.y},{w.id},{w.top},{w.bottom},{w.q}" for w in case.wells]
    ipf.write_text("\n".join(header + rows) + "\n", encoding="ascii")
    for well_id, records in (case.series or {}).items():
        series_header = [str(len(records)), "2,1", "DATE,-999.99", "Q,-999.99"]
        series_rows = [f"{date},{q}" for date, q in records]
        text = "\n".join(series_header + series_rows) + "\n"
        (folder / f"{well_id}.txt").write_text(text, encoding="ascii")

    settings = {"ISS": "0", **case.ini}
    keywords = [
        "FUNCTION=MKWELLIPF",
        "NIPF=1",
        f"IPF1={ipf}",
        "IXCOL=1",
        "IYCOL=2",
        "IQCOL=6",
        "ITCOL=4",
        "IBCOL=5",
        *model_lines,
        *(f"{key}={value}" for key, value in settings.items()),
    ]
    inifile = folder / "mkwellipf.ini"
    inifile.write_text("\n".join(keywords) + "\n", encoding="ascii")
    return inifile


def prepare_case(case: str, work_dir: Path) -> tuple[Path, dict[str, Path]]:
    """Write the layers once and the inputs for each implementation (the ini files)."""
    definition = CASES[case]
    case_dir = work_dir / case
    model_lines = []
    if definition.model:
        model_lines = write_model(case_dir / "model", definition.model)
    inifiles = {
        who: write_inputs(case_dir / who, definition, model_lines)
        for who in ("imod5", "ours")
    }
    return case_dir, inifiles


def result_files(folder: Path) -> dict[str, Path]:
    """The result IPFs below the folder by file name, wherever they were written."""
    return {path.name: path for path in sorted(folder.rglob("imod_mkipf_*.ipf"))}


def read_result_table(path: Path) -> pd.DataFrame:
    """
    Read an IPF as text, without following the associated files: the columns of the
    header and the rows as strings, without quotes. Comma or space separated.
    """
    lines = path.read_text().splitlines()
    ncolumn = int(lines[1])
    names = [line.strip().strip('"') for line in lines[2 : 2 + ncolumn]]
    rows = [line for line in lines[3 + ncolumn :] if line.strip()]
    separator = "," if rows and "," in rows[0] else r"\s+"
    table = pd.read_csv(
        io.StringIO("\n".join(rows)),
        sep=separator,
        header=None,
        names=names,
        dtype=str,
        keep_default_na=False,
        engine="python",
    )
    return table.apply(lambda column: column.str.strip().str.strip('"'))


def compare_tables(
    ours: pd.DataFrame, theirs: pd.DataFrame, nodata: float
) -> list[str]:
    """
    Differences per cell, labelled with the well and layer. Numbers are compared
    within a tolerance, text exactly. Where iMOD5 has no layer value (a well outside
    the grid) ours must have `nodata`, the FNODATA value.
    """
    if [c.lower() for c in ours.columns] != [c.lower() for c in theirs.columns]:
        columns = f"ours {list(ours.columns)}, iMOD5 {list(theirs.columns)}"
        return [f"columns differ: {columns}"]
    if ours.shape != theirs.shape:
        return [f"shape differs: ours {ours.shape}, iMOD5 {theirs.shape}"]

    label_columns = [c for c in ("ID", "ILAY") if c in ours.columns]
    labels = ours[label_columns].apply(" ".join, axis=1)
    differences = []
    for column in ours.columns:
        a, b = ours[column], theirs[column]
        a_number = pd.to_numeric(a, errors="coerce")
        b_number = pd.to_numeric(b, errors="coerce")
        if a_number.notna().all() and b_number.notna().all():
            different = ~np.isclose(a_number, b_number, rtol=RTOL, atol=ATOL)
            if column.upper() in LAYER_COLUMNS:
                no_value = (b_number >= IMOD5_NO_VALUE).to_numpy()
                different = np.where(no_value, a_number.to_numpy() != nodata, different)
        else:
            different = (a != b).to_numpy()
        differences += [
            f"{labels.iloc[i]} {column}: ours {a.iloc[i]}, iMOD5 {b.iloc[i]}"
            for i in np.flatnonzero(different)[:SHOWN]
        ]
    return differences


def compare_results(
    ours: dict[str, Path], theirs: dict[str, Path], nodata: float
) -> tuple[list[str], str]:
    """The differences between the result files, and the tables side by side."""
    differences, listing = [], []
    if set(ours) != set(theirs):
        only_ours = sorted(set(ours) - set(theirs))
        only_imod5 = sorted(set(theirs) - set(ours))
        differences.append(
            f"files differ: only ours {only_ours}, only iMOD5 {only_imod5}"
        )
    for name in sorted(set(ours) & set(theirs)):
        a, b = read_result_table(ours[name]), read_result_table(theirs[name])
        listing += [f"=== {name}: ours", a.to_string()]
        listing += [f"=== {name}: iMOD5", b.to_string(), ""]
        differences += [f"{name}: {d}" for d in compare_tables(a, b, nodata)]
    return differences, "\n".join(listing)


@pytest.fixture(scope="session", autouse=True)
def imod5_works(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Fail once, with diagnostics, when iMOD5 cannot be driven from here."""
    case = Case([Well("probe", -5, -35)], KHV)
    case_dir = tmp_path_factory.mktemp("_mkwellipf_smoke")
    model_lines = write_model(case_dir / "model", KHV)
    inifile = write_inputs(case_dir / "imod5", case, model_lines)
    result = run_imod5(inifile)
    if not result_files(inifile.parent):
        pytest.fail(
            "iMOD5 did not write a result IPF for a single well.\n"
            f"Command: {[str(IMOD5_EXE), str(inifile)]}\n"
            f"Return code: {result.returncode}\n"
            f"stdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}\n"
            f"Run file:\n{inifile.read_text()}\n"
            "Try running that command by hand."
        )


@pytest.mark.parametrize("case", CASES)
def test_mkwellipf_matches_imod5(tmp_path: Path, case: str):
    case_dir, inifiles = prepare_case(case, tmp_path)

    imod5 = run_imod5(inifiles["imod5"])
    mkwellipf(read_mkwellipf_settings(inifiles["ours"]))

    ours = result_files(inifiles["ours"].parent)
    theirs = result_files(inifiles["imod5"].parent)
    nodata = float(CASES[case].ini.get("FNODATA", NODATA))
    differences, listing = compare_results(ours, theirs, nodata)
    has_differences = bool(differences)

    report = (
        f"Results differ, see {case_dir}.\n"
        f"iMOD5 return code: {imod5.returncode} {imod5.stderr[-500:]}\n"
        + "\n".join(differences)
    )
    (case_dir / "comparison.txt").write_text(listing, encoding="utf-8")
    (case_dir / "diff.txt").write_text(
        report if has_differences else "identical\n", encoding="utf-8"
    )

    if case in KNOWN_DIFFERENCES and has_differences:
        pytest.xfail(KNOWN_DIFFERENCES[case])
    assert not has_differences, report
