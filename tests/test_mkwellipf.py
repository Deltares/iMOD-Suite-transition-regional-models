"""Tests for mkwellipf, using a small flat model with three layers.

Layers (top, bottom, K): (0, -10, 10), (-10, -20, 2), (-20, -40, 25). They touch, as in MODFLOW 6.
"""

import logging
from pathlib import Path

import imod
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pydantic import ValidationError

from src.mkwellipf import mkwellipf, well_rates
from src.settings import MkWellIpfSettings

TOP = [0.0, -10.0, -20.0]
BOTTOM = [-10.0, -20.0, -40.0]
KHV = [10.0, 2.0, 25.0]


def write_layers(folder: Path, name: str, values: list[float]) -> list[Path]:
    """Write one IDF per layer on a 4 x 4 grid of 10 m cells, with a constant value."""
    folder.mkdir(exist_ok=True)
    coords = {
        "layer": range(1, len(values) + 1),
        "y": [35.0, 25.0, 15.0, 5.0],
        "x": [5.0, 15.0, 25.0, 35.0],
        "dx": 10.0,
        "dy": -10.0,
    }
    data = np.array(values, dtype=np.float32)[:, None, None] * np.ones((1, 4, 4), np.float32)
    imod.idf.save(folder / name, xr.DataArray(data, coords, ("layer", "y", "x")))
    return [folder / f"{name}_l{i}.idf" for i in range(1, len(values) + 1)]


@pytest.fixture
def model(tmp_path: Path) -> dict:
    folder = tmp_path / "model"
    return {
        "top_files": write_layers(folder, "top", TOP),
        "bottom_files": write_layers(folder, "bot", BOTTOM),
        "khv_files": write_layers(folder, "khv", KHV),
    }


def make_ipf(tmp_path: Path, wells: list[tuple], assoc_col: int = 0) -> Path:
    """Write an IPF with wells (id, screen top, screen bottom, rate), all in the same cell."""
    lines = [str(len(wells)), "6", "X", "Y", "ID", "Z_TOP", "Z_BOT", "Q", f"{assoc_col},txt"]
    lines += [f"15.0,15.0,{wid},{top},{bottom},{q}" for wid, top, bottom, q in wells]
    ipf = tmp_path / "wells.ipf"
    ipf.write_text("\n".join(lines) + "\n")
    return ipf


def run_mkwellipf(ipf: Path, **kwargs) -> list[Path]:
    return mkwellipf(MkWellIpfSettings(ipf_files=[ipf], **kwargs))


def run(tmp_path: Path, wells: list[tuple], **kwargs) -> dict[str, list[float]]:
    """Run mkwellipf and return the fractions per layer of each well."""
    kwargs.setdefault("q_col", 6)
    run_mkwellipf(make_ipf(tmp_path, wells), **kwargs)
    table = imod.ipf.read(tmp_path / "wells" / "imod_mkipf_wells_all.ipf")
    return {
        wid: group["FRACTION"].tolist() for wid, group in table.groupby("ID", sort=False)
    }


def write_series(folder: Path, name: str, records: list[tuple]) -> None:
    """Write a .txt time series (date, rate) with nodata -999.99."""
    lines = [str(len(records)), "2,1", "DATE,-999.99", "Q,-999.99"]
    lines += [f"{date},{q}" for date, q in records]
    (folder / f"{name}.txt").write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize(
    ("iss", "records", "expected"),
    [
        # ISS=0: plain mean, dates and nodata records are ignored
        (0, [(20200102, -10), (20200201, -30), (20200301, -999.99)], -20.0),
        (0, [(20200102, -999.99), (20200201, -999.99)], 0.0),
        (0, [], 0.0),
        # ISS=1, window 1 to 11 January: -10 for 3 days, -20 for 5 days, 0 for 2 days, the
        # record before the window sets the start
        (1, [(20191201, -10), (20200104, -20), (20200109, 0)], -13.0),
        (1, [(20200106, -10)], -5.0),  # 0 before the first record, which counts
        (1, [(20200101, -10), (20200104, -999.99), (20200108, -30)], -20.0),  # inner nodata
        (1, [(20200104, -999.99), (20200108, -20)], -10.0),  # leading nodata: 0 for 3 days
        (1, [(20200101, -10), (20200106, -999.99)], -10.0),  # trailing nodata
        (1, [(20200101, -999.99), (20200104, -999.99)], 0.0),
        (1, [], 0.0),
    ],
)
def test_time_series_average(tmp_path: Path, iss: int, records: list[tuple], expected: float) -> None:
    write_series(tmp_path, "A", records)
    wells = pd.DataFrame({"ID": ["A"]})

    rates = well_rates(
        wells, tmp_path / "wells.ipf", 1, "txt", 1, iss, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-11")
    )

    assert rates == pytest.approx([expected])


def test_rate_is_split_over_layers(tmp_path: Path, model: dict) -> None:
    """Weights are screen length * K * a correction for the screen centre: 25, 20 and 281.25."""
    fractions = run(tmp_path, [("A", -5, -35, -100)], **model)

    assert fractions["A"] == pytest.approx([25 / 326.25, 20 / 326.25, 281.25 / 326.25], abs=1e-4)


def test_minkht_removes_layers_with_low_transmissivity(tmp_path: Path, model: dict) -> None:
    """Layer 2 has K * thickness = 20, below MINKHT. The others share the well."""
    fractions = run(tmp_path, [("A", -5, -35, -100)], minkht=30.0, **model)

    assert fractions["A"] == pytest.approx([25 / 306.25, 0.0, 281.25 / 306.25], abs=1e-4)


@pytest.mark.parametrize(
    ("top", "bottom"),
    [
        (5, 2),  # above the model
        (-70, -80),  # below the model
    ],
)
def test_screens_without_overlap_are_unassigned(tmp_path: Path, model: dict, top: float, bottom: float) -> None:
    fractions = run(tmp_path, [("A", top, bottom, -100)], **model)

    assert fractions["A"] == [0.0, 0.0, 0.0]


def test_point_screen_goes_to_its_layer(tmp_path: Path, model: dict) -> None:
    fractions = run(tmp_path, [("A", -15, -15, -100)], **model)

    assert fractions["A"] == [0.0, 1.0, 0.0]


def test_point_screen_in_layer_below_minkht_is_unassigned(tmp_path: Path, model: dict) -> None:
    """Layer 2 has K * thickness = 20, below MINKHT = 30. iMOD5 would still use layer 2."""
    fractions = run(tmp_path, [("A", -15, -15, -100)], minkht=30.0, **model)

    assert fractions["A"] == [0.0, 0.0, 0.0]


def test_warns_when_layers_do_not_touch(tmp_path: Path, model: dict, caplog: pytest.LogCaptureFixture) -> None:
    gaps = {**model, "top_files": write_layers(tmp_path / "gaps", "top", [0.0, -12.0, -20.0])}

    with caplog.at_level(logging.WARNING, logger="imod"):
        run(tmp_path, [("A", -2, -8, -100)], **gaps)

    assert "assumes layers that touch" in caplog.text


def test_minkd_gives_unscreened_well_to_first_layer_above_transmissivity(tmp_path: Path, model: dict) -> None:
    """Transmissivities 5, 50 and 300 m2/d, MINKD 40: layer 2, as in the iMOD5 manual."""
    model = {**model, "khv_files": None, "kd_files": [5.0, 50.0, 300.0]}

    fractions = run(tmp_path, [("A", -99999, -99999, -100)], minkd=40.0, **model)

    assert fractions["A"] == [0.0, 1.0, 0.0]


def test_unscreened_well_is_unassigned_without_minkd(tmp_path: Path, model: dict) -> None:
    fractions = run(tmp_path, [("A", -99999, -99999, -100)], minkd=40.0, **model)

    assert fractions["A"] == [0.0, 0.0, 0.0]


def test_result_files(tmp_path: Path, model: dict) -> None:
    """Rows per well and layer, the real row count in the header, and no file for layers without wells."""
    ipf = make_ipf(tmp_path, [("A", -2, -8, -100), ("B", -70, -80, -50)])

    written = run_mkwellipf(ipf, q_col=6, **model)

    folder = tmp_path / "wells"
    assert {path.name for path in written} == {
        "imod_mkipf_wells_all.ipf", "imod_mkipf_wells_unassigned.ipf", "imod_mkipf_wells_l1.ipf",
    }
    lines = (folder / "imod_mkipf_wells_all.ipf").read_text().splitlines()
    assert int(lines[0]) == len(lines) - 3 - int(lines[1]) == 6
    table = imod.ipf.read(folder / "imod_mkipf_wells_all.ipf")
    assert table["ERROR_CODE"].tolist() == ["-"] * 3 + ["#"] * 3
    assert table["Q_ORG"].tolist() == [-100] * 3 + [-50] * 3
    assert table["Q_ASSIGNED"].tolist() == [-100.0, 0.0, 0.0] + [0.0] * 3
    unassigned = imod.ipf.read(folder / "imod_mkipf_wells_unassigned.ipf")
    assert unassigned["ID"].tolist() == ["B"] * 3


def test_time_series_with_iss_1(tmp_path: Path, model: dict) -> None:
    """With ISS=1 the column with the file names is replaced, and Q_ORG gets the names."""
    (tmp_path / "A.txt").write_text(
        "3\n2,1\nDATE,-999.99\nQ,-999.99\n20191201,-10\n20200104,-20\n20200109,0\n"
    )
    ipf = make_ipf(tmp_path, [("A", -2, -8, -100)], assoc_col=3)

    run_mkwellipf(ipf, iss=1, sdate="2020-01-01", edate="2020-01-11", **model)

    path = tmp_path / "wells" / "imod_mkipf_wells_all.ipf"
    lines = path.read_text().splitlines()
    assert lines[4] == "Q_ASSIGNED"  # column 3, after the line with the number of rows and columns
    assert lines[2 + int(lines[1])] == "7,txt"  # the file names are now in Q_ORG
    # imod.ipf.read would look for the .txt files, so check the data lines themselves
    rows = [line.split(",") for line in lines[3 + int(lines[1]) :]]
    assert [row[6].strip('"') for row in rows] == ["A"] * 3  # Q_ORG
    assert [float(row[2]) for row in rows] == [-13.0, 0.0, 0.0]  # Q_ASSIGNED


def test_without_layers_only_the_rate_is_written(tmp_path: Path) -> None:
    ipf = make_ipf(tmp_path, [("A", -2, -8, -100)])

    written = run_mkwellipf(ipf, q_col=6)

    table = imod.ipf.read(written[0])
    assert written[0].name == "imod_mkipf_wells.ipf"
    assert table["Q_ASSIGNED"].tolist() == [-100.0]
    assert table["Q_ORG"].tolist() == [-100]


def test_space_separated_ipf(tmp_path: Path) -> None:
    ipf = make_ipf(tmp_path, [("A", -2, -8, -100)])
    lines = ipf.read_text().splitlines()
    ipf.write_text("\n".join(lines[:9] + [line.replace(",", " ") for line in lines[9:]]) + "\n")

    written = run_mkwellipf(ipf, q_col=6)

    table = imod.ipf.read(written[0])
    assert table["ID"].tolist() == ["A"]
    assert table["Q_ASSIGNED"].tolist() == [-100.0]


def test_existing_result_requires_overwrite(tmp_path: Path, model: dict) -> None:
    ipf = make_ipf(tmp_path, [("A", -2, -8, -100)])
    run_mkwellipf(ipf, q_col=6, **model)

    with pytest.raises(FileExistsError):
        run_mkwellipf(ipf, q_col=6, **model)

    run_mkwellipf(ipf, q_col=6, overwrite=True, **model)


def test_settings_iss_1_requires_dates(tmp_path: Path) -> None:
    ipf = make_ipf(tmp_path, [("A", -2, -8, -100)])

    with pytest.raises(ValidationError, match="SDATE and EDATE are required"):
        MkWellIpfSettings(ipf_files=[ipf], iss=1)


def test_settings_khv_and_kd_are_exclusive(tmp_path: Path, model: dict) -> None:
    ipf = make_ipf(tmp_path, [("A", -2, -8, -100)])

    with pytest.raises(ValidationError, match="not both"):
        MkWellIpfSettings(ipf_files=[ipf], kd_files=[1.0, 1.0, 1.0], **model)


def test_settings_top_and_bottom_need_the_same_number_of_layers(tmp_path: Path, model: dict) -> None:
    ipf = make_ipf(tmp_path, [("A", -2, -8, -100)])

    with pytest.raises(ValidationError, match="same number of layers"):
        MkWellIpfSettings(ipf_files=[ipf], top_files=model["top_files"])
