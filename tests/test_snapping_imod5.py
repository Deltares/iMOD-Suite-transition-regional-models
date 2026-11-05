"""
Compare snapping with the GENSNAPTOGRID function of iMOD5.

Meant to be run locally: the tests are skipped when the iMOD5 executable does not
exist. Set the ``IMOD5_EXE`` environment variable to its path.

Synthetic GEN files and IDFs are written to the pytest tmp folder, snapped by iMOD5
and by ``snap_gen_file_to_grid``, and the results are compared. For ``I3D=0`` that
is the set of walls (iMOD5 writes one line per wall, we write merged lines); for
``I3D=1`` the vertical walls (polygons with Z), in the order of iMOD5 and with the
ring vertex order. The input, both outputs, the ini file and the console output of
iMOD5 are kept in the folder of the case. Edge selection uses xugrid; named
legacy differences are asserted explicitly, with common 3D rings still matching.

The cases (``CASES``) run on a 4 x 4 grid of cells of size 1 with its origin at (0, 0),
with I3D=0 and with I3D=1. The coded TOP value of a cell is ``1000 + 10 * col + row``
and the BOT value is ``TOP - 500``, with ``col`` and ``row`` counted from zero at the
left and at the bottom, so the Z values of a wall tell which cell(s) iMOD5 sampled.
iMOD5 rounds some single precision Z values differently to 3 decimals than we do:
Z values are compared with a tolerance of 0.001.
"""

import itertools
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import imod
import numpy as np
import pytest
import shapely
import xarray as xr

from src.grid_helpers import grid_spec, open_idf_grid
from src.snapping import snap_gen_file_to_grid

IMOD5_EXE = Path(
    os.environ.get("IMOD5_EXE", r"C:\Development\bin\iMOD5\iMOD 5.6.1\iMOD_V5_6_1.exe")
)
pytestmark = [
    pytest.mark.imod5,
    pytest.mark.skipif(
        not IMOD5_EXE.exists(),
        reason=f"{IMOD5_EXE} not found. Set the IMOD5_EXE environment variable.",
    ),
]

Coord = tuple[float, float] # (x, y)
Line = list[Coord]
Window = tuple[float, float, float, float] # (xmin, ymin, xmax, ymax)
Grid = dict  # IDFFILE, WINDOW and CELL_SIZE as keyword arguments of run_imod5
Ring = tuple[tuple[float, float, float], ...] # (x, y, z) vertices of a polygon ring

@dataclass(frozen=True)
class Case:
    """
    A test case that is run with I3D=0 and with I3D=1, on a 4 x 4 grid of cells of
    size 1 with its origin at (0, 0).

    lines: features, in coordinates of the grid.
    ids: ids of the features in the GEN file; their positions if None.
    grid: where iMOD5 gets its grid from: ``window`` (WINDOW and CELL_SIZE),
        ``idffile`` (IDFFILE only) or ``idffile-window`` (IDFFILE, WINDOW and
        CELL_SIZE). (IDFFILE with WINDOW but without CELL_SIZE makes iMOD5 open a
        dialog and write nothing.)
    kind: how the TOP and BOT IDFs are built, see ``write_case_idfs``.
    nodata_cells: (col, row) of cells set to nodata in the IDF(s) of ``nodata_in``
        (``both``, ``top`` or ``bot``).
    """

    lines: list[Line]
    ids: list[str] | None = None
    grid: str = "window"
    kind: str = "coded"
    nodata_cells: tuple[tuple[int, int], ...] = ()
    nodata_in: str = "both"

    @property
    def varies_values(self) -> bool:
        """Only the TOP and BOT values differ from the default: no use with I3D=0."""
        return self.kind != "coded" or bool(self.nodata_cells)


IDF_WINDOW: Window = (1.0, 0.0, 3.0, 4.0)
HORIZONTAL: Line = [(0.1, 1.1), (3.9, 1.1)]
VERTICAL: Line = [(1.1, 0.1), (1.1, 3.9)]

CASES: dict[str, Case] = {
    "horizontal": Case([HORIZONTAL]),
    "vertical": Case([VERTICAL]),
    "diagonal-short": Case([[(1.1, 1.1), (2.9, 2.9)]]),
    "diagonal-full": Case([[(0.1, 0.1), (3.9, 3.9)]]),
    "through-nodes": Case([[(1.0, 1.0), (3.0, 3.0)]]),
    "shallow-slope": Case([[(0.2, 0.3), (3.8, 1.4)]]),
    "polyline": Case([[(0.6, 1.4), (2.6, 1.4), (2.6, 3.4)]]),
    "crossing-window": Case([[(-1.0, 1.1), (5.0, 1.1)]]),
    "reentering": Case([[(1.0, 1.0), (5.0, 1.0), (5.0, 3.0), (1.0, 3.0)]]),
    "along-gridline": Case([[(0.0, 2.0), (4.0, 2.0)]]),
    "domain-boundary": Case([[(0.0, 0.0), (4.0, 0.0)]]),
    "boundary-right": Case([[(4.0, 0.0), (4.0, 4.0)]]),
    "boundary-then-inside": Case([[(0.0, 0.0), (0.0, 2.0), (2.0, 2.0)]]),
    "short-in-cell": Case([[(1.01, 1.01), (1.02, 1.02)]]),
    "outside": Case([[(10.0, 10.0), (12.0, 12.0)]]),
    "two-lines-same-edge": Case(
        [HORIZONTAL, [(0.2, 1.2), (3.8, 1.2)]], ids=["a", "b"]
    ),
    # How line ends are treated: ends 0.2 before / after the cell centre, reversed
    # lines, in x, and slanted.
    "end-before-centre": Case([[(1.1, 0.1), (1.1, 2.3)]]),
    "end-after-centre": Case([[(1.1, 0.1), (1.1, 2.7)]]),
    "start-before-centre": Case([[(1.1, 1.3), (1.1, 3.9)]]),
    "reversed-end-before-centre": Case([[(1.1, 2.3), (1.1, 0.1)]]),
    "end-x-before-centre": Case([[(0.1, 1.1), (2.3, 1.1)]]),
    "slanted-end-before-centre": Case([[(1.1, 0.1), (1.3, 2.3)]]),
    # Only the end of the last feature that reaches the grid is extended.
    "first-feature-end-not-extended": Case(
        [[(1.1, 0.1), (1.1, 2.3)], [(2.1, 0.1), (2.1, 3.9)]]
    ),
    "last-feature-end-extended": Case(
        [[(2.1, 0.1), (2.1, 3.9)], [(1.1, 0.1), (1.1, 2.3)]]
    ),
    "last-feature-outside-ignored": Case(
        [[(1.1, 0.1), (1.1, 2.3)], [(10.0, 10.0), (12.0, 12.0)]]
    ),
    "last-feature-in-single-cell": Case(
        [[(1.1, 0.1), (1.1, 2.3)], [(3.1, 3.1), (3.3, 3.2)]]
    ),
    "last-feature-one-piece-other-cell": Case(
        [[(0.5, 0.5), (3.5, 0.5)], [(2.2, 2.2), (2.4, 2.2)]]
    ),
    # Lines through cell centres, and on grid lines, half-cell lines and nodes:
    # the exact comparisons of INTERSECT_EQUI.
    "tie-diagonal": Case([[(1.5, 1.5), (2.5, 2.5)]]),
    "on-half-cell-vertical": Case([[(0.5, 0.0), (0.5, 4.0)]]),
    "on-half-cell-diagonal": Case([[(0.25, 0.25), (3.75, 3.75)]]),
    "on-node-then-half-cell": Case(
        [[(1.0, 1.0), (1.5, 1.0), (1.5, 2.5), (3.0, 2.5)]]
    ),
    # Duplicate vertices.
    "duplicate-vertices": Case([[(1.1, 1.1), (1.1, 1.1), (2.9, 2.9)]]),
    "duplicate-end-vertices": Case([[(0.6, 1.4), (2.6, 1.4), (2.6, 1.4)]]),
    "last-feature-duplicate-points": Case(
        [[(1.1, 0.1), (1.1, 2.3)], [(2.2, 2.2), (2.2, 2.2), (2.4, 2.2)]]
    ),
    # Lines that start outside, or leave and re-enter the grid.
    "leave-and-reenter-top": Case(
        [[(1.5, 1.5), (1.5, 5.0), (3.5, 5.0), (3.5, 1.5)]]
    ),
    "start-outside-diagonal": Case([[(-1.0, -1.0), (3.0, 2.0)]]),
    # Grids.
    "idffile-grid": Case([HORIZONTAL], grid="idffile"),
    "idffile-window": Case([HORIZONTAL], grid="idffile-window"),
    # Which cell does TOP / BOT come from? With values decreasing in x and y,
    # "right/upper cell" and "highest value" no longer coincide.
    "reversed-vertical": Case([VERTICAL], kind="reversed"),
    "reversed-horizontal": Case([HORIZONTAL], kind="reversed"),
    # Nodata: either side of a vertical or horizontal edge, in one or both IDFs.
    "nodata-right": Case([VERTICAL], nodata_cells=((1, 1),)),
    "nodata-both-sides": Case([VERTICAL], nodata_cells=((1, 1), (0, 1))),
    "nodata-left": Case([VERTICAL], nodata_cells=((0, 1),)),
    "nodata-upper": Case([HORIZONTAL], nodata_cells=((1, 1),)),
    "nodata-lower": Case([HORIZONTAL], nodata_cells=((1, 0),)),
    "nodata-top-only": Case([VERTICAL], nodata_cells=((1, 1),), nodata_in="top"),
    "nodata-bot-only": Case([VERTICAL], nodata_cells=((1, 1),), nodata_in="bot"),
    # TOP below, equal to, and constant equal to BOT.
    "top-below-bot": Case([VERTICAL], kind="top-below-bot"),
    "top-equals-bot": Case([VERTICAL], kind="top-equals-bot"),
    "constant-equal": Case([VERTICAL], kind="constant-equal"),
}

# name: (xmin, ymin, cellsize, ncols, nrows)
GRIDS = {
    "unit": (0.0, 0.0, 1.0, 4, 4),
    "rd": (100_000.0, 400_000.0, 100.0, 4, 4),
    "large": (175_000.0, 364_000.0, 25.0, 1040, 1320),
    "random": (0.0, 0.0, 1.0, 12, 12),
}

# Cases that only differ in the TOP and BOT values are the same with I3D=0.
GEOMETRY_CASES = [name for name, case in CASES.items() if not case.varies_values]

# Cell widths and heights of a non-equidistant IDF; origin at (0, 0).
NONUNIFORM_DX = [1.0, 2.0, 0.5, 1.5, 1.0, 2.0]
NONUNIFORM_DY = [1.0, 0.5, 2.0, 1.0, 1.5]
NONUNIFORM_CASES = {
    "horizontal": [[(0.1, 1.1), (7.9, 1.1)]],
    "diagonal": [[(0.3, 0.2), (7.7, 5.8)]],
    "polyline": [[(0.6, 1.4), (4.6, 1.4), (4.6, 5.4), (2.2, 3.1)]],
    "leaving": [[(-1.0, 2.2), (9.0, 2.2), (9.0, 4.4), (3.0, 4.4)]],
}

def grid_window(name: str) -> tuple[Window, float]:
    xmin, ymin, cellsize, ncols, nrows = GRIDS[name]
    return (xmin, ymin, xmin + ncols * cellsize, ymin + nrows * cellsize), cellsize


def in_coordinates(lines: list[Line], name: str) -> list[Line]:
    """Lines in units of cells as coordinates on the grid ``name``."""
    xmin, ymin, cellsize, _, _ = GRIDS[name]
    return [[(xmin + x * cellsize, ymin + y * cellsize) for x, y in line] for line in lines]


def make_meandering_lines(ncols: int, nrows: int, rng: np.random.Generator) -> list[Line]:
    """
    Synthetic stand-in for a real-world GEN file, in units of cells: meandering
    watercourses with many vertices, cut off where they leave the grid, and one
    long straight line (e.g. a dike) crossing the whole grid.
    """
    lines: list[Line] = []
    for _ in range(25):
        x, y = rng.uniform(0.0, ncols), rng.uniform(0.0, nrows)
        heading = rng.uniform(0.0, 2.0 * np.pi)
        line: Line = [(x, y)]
        for _ in range(rng.integers(100, 400)):
            heading += rng.normal(0.0, 0.4)
            step = rng.uniform(0.2, 6.0)
            x += step * np.cos(heading)
            y += step * np.sin(heading)
            if not (0.0 <= x <= ncols and 0.0 <= y <= nrows):
                break
            line.append((x, y))
        if len(line) > 1:
            lines.append(line)
    lines.append([(-20.0, 49.38), (ncols + 20.0, nrows - 172.868)])
    return lines


def make_random_lines(ncols: int, nrows: int, n_lines: int, rng: np.random.Generator, margin: int = 2) -> list[Line]:
    """
    Random polylines in units of cells with vertices on multiples of 0.125, so
    that vertices and segments often lie exactly on grid lines, half-cell lines
    and nodes. They reach ``margin`` cells outside the grid. Consecutive
    duplicate vertices occur.
    """
    lines: list[Line] = []
    while len(lines) < n_lines:
        n_vertices = int(rng.integers(2, 7))
        x = rng.integers(-8 * margin, 8 * (ncols + margin) + 1, size=n_vertices) / 8.0
        y = rng.integers(-8 * margin, 8 * (nrows + margin) + 1, size=n_vertices) / 8.0
        line = list(zip(x.tolist(), y.tolist()))
        # GEN readers take a closed line for a polygon.
        if line[0] != line[-1]:
            lines.append(line)
    return lines


# Running both implementations


def write_ascii_gen(path: Path, lines: list[Line], ids: list[str] | None = None) -> None:
    ids = ids or [str(number) for number in range(1, len(lines) + 1)]
    text = []
    for feature_id, coordinates in zip(ids, lines):
        text.append(feature_id)
        text.extend(f"{x:.6f},{y:.6f}" for x, y in coordinates)
        text.append("END")
    text.append("END")
    path.write_text("\n".join(text) + "\n", encoding="ascii")


def run_imod5(
    case_dir: Path,
    gen_in: Path,
    gen_out: Path,
    *,
    window: Window | None = None,
    cellsize: float | None = None,
    idffile: Path | None = None,
    idf_top: Path | None = None,
    idf_bot: Path | None = None,
) -> subprocess.CompletedProcess:
    gen_out.unlink(missing_ok=True)
    lines = ["FUNCTION=GENSNAPTOGRID", f"GENFILE_IN={gen_in}", f"GENFILE_OUT={gen_out}"]
    if idffile is not None:
        lines.append(f"IDFFILE={idffile}")
    if window is not None:
        lines.append(f"WINDOW={','.join(f'{value:.3f}' for value in window)}")
    if cellsize is not None:
        lines.append(f"CELL_SIZE={cellsize}")
    if idf_top is not None:
        lines += ["I3D=1", f"IDF_TOP={idf_top}", f"IDF_BOT={idf_bot}"]
    inifile = case_dir / "snap.ini"
    inifile.write_text("\n".join([*lines, ""]), encoding="ascii")
    result = subprocess.run(
        [str(IMOD5_EXE), str(inifile)],
        cwd=case_dir,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
        check=False,
    )
    (case_dir / "imod5_console.txt").write_text(
        f"returncode: {result.returncode}\n--- stdout\n{result.stdout}\n--- stderr\n{result.stderr}\n",
        encoding="utf-8",
    )
    return result


@pytest.fixture(scope="session", autouse=True)
def imod5_works(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Fail once, with diagnostics, when iMOD5 cannot be driven from here."""
    case_dir = tmp_path_factory.mktemp("_smoke")
    write_ascii_gen(case_dir / "input.gen", CASES["horizontal"].lines)
    result = run_imod5(
        case_dir, case_dir / "input.gen", case_dir / "imod5.gen", window=(0.0, 0.0, 4.0, 4.0), cellsize=1.0
    )
    assert (case_dir / "imod5.gen").exists(), (
        "iMOD5 did not write an output GEN for a simple horizontal line.\n"
        f"Return code: {result.returncode}\nstdout: {result.stdout[-1000:]}\n"
        f"stderr: {result.stderr[-1000:]}\nIni file:\n{(case_dir / 'snap.ini').read_text()}"
    )

@pytest.fixture(scope="session")
def session_rng() -> np.random.Generator:
    return np.random.default_rng(42)

def run_both(
    case_dir: Path,
    lines: list[Line],
    grid: Grid,
    ids: list[str] | None = None,
    top: Path | None = None,
    bot: Path | None = None,
) -> tuple[Path, Path]:
    """Snap with ours and with iMOD5; returns the paths of both output files."""
    case_dir.mkdir(parents=True, exist_ok=True)
    gen_in, ours, theirs = case_dir / "input.gen", case_dir / "ours.gen", case_dir / "imod5.gen"
    write_ascii_gen(gen_in, lines, ids)
    run_imod5(case_dir, gen_in, theirs, idf_top=top, idf_bot=bot, **grid)

    # The cell size of an IDF is the one of its grid, also with a window.
    idffile = grid.get("idffile")
    ours.unlink(missing_ok=True)
    try:
        snap_gen_file_to_grid(
            gen_in,
            ours,
            like=None if idffile is None else open_idf_grid(idffile),
            window=grid.get("window"),
            cellsize=None if idffile is not None else grid.get("cellsize"),
            idf_top=top,
            idf_bot=bot,
        )
    except ValueError as e:
        if "No line features" not in str(e):
            raise
    return ours, theirs


# I3D=0


def read_gen_or_empty(path: Path) -> gpd.GeoDataFrame:
    """iMOD5 writes an empty result as a file containing only ``END``."""
    if not path.exists() or path.read_bytes().strip().upper() == b"END":
        return gpd.GeoDataFrame(geometry=[])
    return imod.gen.read(path)


def wall_set(gdf: gpd.GeoDataFrame) -> set[frozenset]:
    """The walls of (merged) lines as unordered pairs of end points, to 3 decimals."""
    walls = set()
    for part in shapely.get_parts(gdf.geometry.to_numpy()):
        xy = np.round(np.asarray(part.coords)[:, :2], 3)
        walls |= {frozenset({tuple(a), tuple(b)}) for a, b in itertools.pairwise(xy)}
    return walls

# Accepted deltas for known iMOD5 differences in wall counts
# id: (only_ours, only_imod5)
ACCEPTED_DELTAS = {
    "polyline": (1, 0),
    "end-before-centre": (1, 0),
    "end-x-before-centre": (1, 0),
    "slanted-end-before-centre": (1, 0),
    "last-feature-end-extended": (1, 0),
    "last-feature-outside-ignored": (1, 0),
    "last-feature-one-piece-other-cell": (0, 2),
    "tie-diagonal": (0, 3),
    "on-half-cell-vertical": (3, 4),
    "on-node-then-half-cell": (2, 2),
    "leave-and-reenter-top": (7, 1),
}


def compare_2d(
    case_dir: Path, lines: list[Line], grid: Grid, expected_delta=(0, 0)
) -> set[frozenset]:
    """Assert known (legacy-only, xugrid-only) wall counts; return legacy walls."""
    ours, theirs = (read_gen_or_empty(p) for p in run_both(case_dir, lines, grid))
    ours_walls, theirs_walls = wall_set(ours), wall_set(theirs)
    only_ours = sorted(tuple(sorted(w)) for w in ours_walls - theirs_walls)
    only_theirs = sorted(tuple(sorted(w)) for w in theirs_walls - ours_walls)
    assert (len(only_theirs), len(only_ours)) == expected_delta, (
        f"Walls differ (files in {case_dir}): ours {len(ours_walls)}, iMOD5 "
        f"{len(theirs_walls)}\nOnly ours: {only_ours[:5]}\nOnly iMOD5: {only_theirs[:5]}"
    )
    return theirs_walls


def _grid_arguments(name: str) -> Grid:
    window, cellsize = grid_window(name)
    return {"window": window, "cellsize": cellsize}


def case_grid(case: Case, top: Path) -> Grid:
    return {
        "window": _grid_arguments("unit"),
        "idffile": {"idffile": top},
        "idffile-window": {"idffile": top, "window": IDF_WINDOW, "cellsize": 1.0},
    }[case.grid]


@pytest.mark.parametrize("name", GEOMETRY_CASES)
def test_snap_has_known_imod5_differences(tmp_path, name):
    case = CASES[name]
    top, _ = write_case_idfs(tmp_path, case)
    compare_2d(tmp_path, case.lines, case_grid(case, top), ACCEPTED_DELTAS.get(name, (0, 0)))


@pytest.mark.parametrize("name", ["horizontal", "polyline", "diagonal-short"])
def test_snap_matches_imod5_on_a_grid_far_from_the_origin(tmp_path, name):
    lines = in_coordinates(CASES[name].lines, "rd")
    delta = {"horizontal": (0, 0), "polyline": (1, 0), "diagonal-short": (2, 2)}[name]
    compare_2d(tmp_path, lines, _grid_arguments("rd"), delta)


def test_window_that_is_no_multiple_of_the_cellsize_matches_imod5(tmp_path):
    # iMOD5 drops the remainder at the upper-right.
    grid = {"window": (0.0, 0.0, 4.6, 4.6), "cellsize": 1.0}
    assert compare_2d(tmp_path, CASES["polyline"].lines, grid, (1, 0))


def test_meandering_lines_on_a_large_grid_match_imod5(tmp_path, session_rng):
    """A realistic size: 1040 x 1320 cells of 25 m with meandering lines."""
    lines = in_coordinates(make_meandering_lines(*GRIDS["large"][3:], rng=session_rng), "large")
    assert compare_2d(tmp_path, lines, _grid_arguments("large"))


@pytest.mark.parametrize("n_lines", [200, 300, 400, 500, 600])
def test_random_polylines_match_imod5(tmp_path, n_lines):
    """Vertices on multiples of 0.125 cells hit the exact-comparison branches."""
    lines = make_random_lines(*GRIDS["random"][3:], n_lines=n_lines, rng=np.random.default_rng(42))
    compare_2d(tmp_path, lines, _grid_arguments("random"))


def _save_idf(case_dir: Path, name: str, da: xr.DataArray) -> Path:
    case_dir.mkdir(parents=True, exist_ok=True)
    for old in case_dir.glob(f"{name}*.idf"):
        old.unlink()
    imod.idf.save(case_dir / name, da.transpose("y", "x").astype(np.float32))
    (path,) = case_dir.glob(f"{name}*.idf")
    return path


def test_idf_header_with_many_digits_matches_imod5(tmp_path):
    """iMOD5 reads ymax 462821.75 of the IDF header as 462821.8."""
    grid = imod.util.empty_2d(
        dx=6.25, xmin=150000.0, xmax=150018.75, dy=-6.25, ymin=462753.0, ymax=462821.75
    )
    idf = _save_idf(tmp_path, "grid", grid)
    lines = [[(150001.0, 462755.0), (150016.0, 462820.0)]]
    assert compare_2d(tmp_path, lines, {"idffile": idf})


@pytest.mark.parametrize("case", NONUNIFORM_CASES)
def test_nonuniform_idf_matches_imod5(tmp_path, case):
    """Pin the measured staircase exchanges instead of retaining port quirks."""
    like = imod.util.empty_2d(
        dx=np.array(NONUNIFORM_DX),
        xmin=0.0,
        xmax=sum(NONUNIFORM_DX),
        dy=-np.array(NONUNIFORM_DY[::-1]),
        ymin=0.0,
        ymax=sum(NONUNIFORM_DY),
    )
    idf = _save_idf(tmp_path, "grid", like * 0.0 + 1.0)
    assert not grid_spec(open_idf_grid(idf)).equidistant
    delta = {"horizontal": (0, 0), "diagonal": (3, 3), "polyline": (1, 1), "leaving": (1, 1)}[case]
    assert compare_2d(tmp_path, NONUNIFORM_CASES[case], {"idffile": idf}, delta)


# I3D=1


def coded_top(cellsize: float = 1.0, reversed_: bool = False) -> xr.DataArray:
    """TOP of the 4 x 4 grid: ``1000 + 10 * col + row``, coded per cell of size 1."""
    template = imod.util.empty_2d(dx=cellsize, xmin=0.0, xmax=4.0, dy=-cellsize, ymin=0.0, ymax=4.0)
    col, row = np.floor(template["x"]), np.floor(template["y"])
    top = 1100.0 - 10.0 * col - row if reversed_ else 1000.0 + 10.0 * col + row
    return top.transpose("y", "x")


def write_case_idfs(case_dir: Path, case: Case | None = None) -> tuple[Path, Path]:
    """
    Write ``top.idf`` and ``bot.idf`` for a case and return their paths.

    kinds: ``coded`` (BOT is ``TOP - 500``); ``reversed`` (TOP is
    ``1100 - 10 * col - row``); ``top-below-bot`` (TOP and BOT swapped in cell
    col 1, row 1); ``top-equals-bot``; ``constant-equal`` (TOP and BOT are 1000
    everywhere); ``finer`` (cells of size 0.5 holding the value of the cell of
    size 1 they lie in).
    """
    case = case or Case([])
    top = coded_top(0.5 if case.kind == "finer" else 1.0, case.kind == "reversed")
    bot = top - 500.0

    def cell(col: int, row: int) -> xr.DataArray:
        x, y = top["x"], top["y"]
        return ((x >= col) & (x < col + 1) & (y >= row) & (y < row + 1)).transpose("y", "x")

    if case.kind == "top-below-bot":
        top, bot = xr.where(cell(1, 1), bot, top), xr.where(cell(1, 1), top, bot)
    elif case.kind == "top-equals-bot":
        bot = top.copy()
    elif case.kind == "constant-equal":
        top = xr.full_like(top, 1000.0)
        bot = top.copy()
    for col, row in case.nodata_cells:
        if case.nodata_in in ("both", "top"):
            top = top.where(~cell(col, row))
        if case.nodata_in in ("both", "bot"):
            bot = bot.where(~cell(col, row))
    return _save_idf(case_dir, "top", top), _save_idf(case_dir, "bot", bot)




def read_walls(path: Path) -> list[tuple[str, Ring]]:
    """
    Read a 3D ASCII GEN as a list of ``(id, ring)``, with the vertices rounded to
    the 3 decimals of the files. Empty for a missing file or one with only ``END``.
    """
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    walls = []
    i = 0
    while i < len(lines) and lines[i].upper() != "END":
        feature_id = lines[i]
        i += 1
        ring = []
        while lines[i].upper() != "END":
            ring.append(tuple(round(float(value), 3) for value in lines[i].split(",")))
            i += 1
        i += 1
        walls.append((feature_id, tuple(ring)))
    return walls


def _same_ring(ours: Ring, theirs: Ring) -> bool:
    return len(ours) == len(theirs) and all(
        a[:2] == b[:2] and abs(a[2] - b[2]) <= 0.0011 for a, b in zip(ours, theirs)
    )


def compare_3d(
    case_dir: Path, lines: list[Line], grid: Grid, top: Path, bot: Path,
    ids: list[str] | None = None, expected_delta=(0, 0)
) -> list:
    """Pin selected-edge deltas and common ring values, ordering and sequential ids."""
    ours_path, theirs_path = run_both(case_dir, lines, grid, ids, top, bot)
    ours, theirs = read_walls(ours_path), read_walls(theirs_path)
    def key(wall):
        return frozenset(vertex[:2] for vertex in wall[1])

    ours_keys, theirs_keys = set(map(key, ours)), set(map(key, theirs))
    assert (len(theirs_keys - ours_keys), len(ours_keys - theirs_keys)) == expected_delta
    shared = ours_keys & theirs_keys
    ours_shared = [wall for wall in ours if key(wall) in shared]
    theirs_shared = [wall for wall in theirs if key(wall) in shared]
    assert len(ours_shared) == len(theirs_shared)
    assert all(_same_ring(o[1], t[1]) for o, t in zip(ours_shared, theirs_shared))
    for walls in (ours, theirs):
        assert [w[0] for w in walls] == [str(n) for n in range(1, len(walls) + 1)]
    return theirs


@pytest.mark.parametrize("name", CASES)
def test_walls_have_known_imod5_differences(tmp_path, name):
    case = CASES[name]
    top, bot = write_case_idfs(tmp_path, case)
    compare_3d(
        tmp_path, case.lines, case_grid(case, top), top, bot, case.ids,
        ACCEPTED_DELTAS.get(name, (0, 0)),
    )


def test_finer_top_and_bottom_are_not_supported(tmp_path):
    """iMOD5 block-averages finer TOP and BOT grids onto the target; we refuse."""
    case = Case([HORIZONTAL], kind="finer")
    top, bot = write_case_idfs(tmp_path, case)
    ours, theirs = tmp_path / "ours.gen", tmp_path / "imod5.gen"
    write_ascii_gen(tmp_path / "input.gen", case.lines)
    run_imod5(tmp_path, tmp_path / "input.gen", theirs, window=(0.0, 0.0, 4.0, 4.0), cellsize=1.0, idf_top=top, idf_bot=bot)
    assert read_walls(theirs)

    with pytest.raises(ValueError, match="same grid"):
        snap_gen_file_to_grid(
            tmp_path / "input.gen", ours, window=(0.0, 0.0, 4.0, 4.0), cellsize=1.0, idf_top=top, idf_bot=bot
        )
def _compare_3d_on_grid(case_dir: Path, lines: list[Line], name: str, top_values, bot_values) -> None:
    top, bot = _save_idf(case_dir, "top", top_values), _save_idf(case_dir, "bot", bot_values)
    assert compare_3d(case_dir, lines, _grid_arguments(name), top, bot)


def _cell_indices(name: str) -> tuple[xr.DataArray, xr.DataArray]:
    (xmin, ymin, xmax, ymax), cellsize = grid_window(name)
    grid = imod.util.empty_2d(dx=cellsize, xmin=xmin, xmax=xmax, dy=-cellsize, ymin=ymin, ymax=ymax)
    return np.floor((grid["x"] - xmin) / cellsize), np.floor((grid["y"] - ymin) / cellsize)


def test_walls_on_a_large_grid_match_imod5(tmp_path, session_rng):
    """Values such as 10.001 are not exact in single precision."""
    cells = make_meandering_lines(*GRIDS["large"][3:], rng=session_rng)
    lines = in_coordinates(cells[:8] + cells[-1:], "large")
    col, row = _cell_indices("large")
    top = 10.0 + 0.001 * col + 0.002 * row
    _compare_3d_on_grid(tmp_path, lines, "large", top, top - 100.0 + 5.0 * np.sin(0.1 * col))


@pytest.mark.parametrize("n_lines", [200, 300, 400, 500, 600])
def test_walls_of_random_polylines_match_imod5(tmp_path, n_lines):
    lines = make_random_lines(*GRIDS["random"][3:], rng=np.random.default_rng(42), n_lines=n_lines)
    col, row = _cell_indices("random")
    _compare_3d_on_grid(tmp_path, lines, "random", 1000.0 + 10.0 * col + row, 500.0 + 10.0 * col - 3.0 * row)
