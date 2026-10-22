"""
Compare snapping with the GENSNAPTOGRID function of iMOD5.

Synthetic GEN files are written to ``pytest_tmp_path/<grid>_<case>/``,
snapped by both implementations, and compared as sets of grid edges. Vertex
order, line direction, and merging into polylines therefore do not matter. The
input, iMOD5 output, our output, ini file, and iMOD5 console output are kept in
that folder for inspection, together with a text diff and a comparison plot
(green: input, red: ours, blue dashed: iMOD5).

Our snapper and iMOD5 are each run once per case. Behaviour established with
these cases is described in the docstring of ``snapping.snap_gen_to_grid``.

Environment variables:

* ``IMOD5_EXE``: path to the iMOD5 executable.

The tests are skipped when the executable does not exist.
"""

import os
import subprocess
from functools import cache
from pathlib import Path

import geopandas as gpd
import imod
import numpy as np
import pytest
import shapely
from matplotlib.figure import Figure

from snapping import snap_gen_file_to_grid

IMOD5_EXE = Path(
    os.environ.get(
        "IMOD5_EXE", r"C:\Development\bin\iMOD5\iMOD 5.6.1\iMOD_V5_6_1.exe"
    )
)
TIMEOUT_SECONDS = 120

pytestmark = [
    pytest.mark.imod5,
    pytest.mark.skipif(not IMOD5_EXE.exists(), reason=f"{IMOD5_EXE} not found"),
]

# Lines in units of cells on a 4 x 4 grid, with the origin at (0, 0).
CASES = {
    "horizontal": [[(0.1, 1.1), (3.9, 1.1)]],
    "vertical": [[(1.1, 0.1), (1.1, 3.9)]],
    "diagonal-short": [[(1.1, 1.1), (2.9, 2.9)]],
    "diagonal-full": [[(0.1, 0.1), (3.9, 3.9)]],
    "through-nodes": [[(1.0, 1.0), (3.0, 3.0)]],
    "shallow-slope": [[(0.2, 0.3), (3.8, 1.4)]],
    "polyline": [[(0.6, 1.4), (2.6, 1.4), (2.6, 3.4)]],
    "crossing-window": [[(-1.0, 1.1), (5.0, 1.1)]],
    "reentering": [[(1.0, 1.0), (5.0, 1.0), (5.0, 3.0), (1.0, 3.0)]],
    "along-gridline": [[(0.0, 2.0), (4.0, 2.0)]],
    "domain-boundary": [[(0.0, 0.0), (4.0, 0.0)]],
    "short-in-cell": [[(1.01, 1.01), (1.02, 1.02)]],
    "outside": [[(10.0, 10.0), (12.0, 12.0)]],
    "two-lines-same-edge": [
        [(0.1, 1.1), (3.9, 1.1)],
        [(0.2, 1.2), (3.8, 1.2)],
    ],
    # Probes for how line ends are treated: vertical lines at x=1.1 whose end
    # (or start) lies 0.2 / 0.05 before / 0.05 / 0.2 after the cell centre.
    "end-before-centre-0.2": [[(1.1, 0.1), (1.1, 2.3)]],
    "end-before-centre-0.05": [[(1.1, 0.1), (1.1, 2.45)]],
    "end-after-centre-0.05": [[(1.1, 0.1), (1.1, 2.55)]],
    "end-after-centre-0.2": [[(1.1, 0.1), (1.1, 2.7)]],
    "start-before-centre-0.2": [[(1.1, 1.3), (1.1, 3.9)]],
    "start-before-centre-0.05": [[(1.1, 1.45), (1.1, 3.9)]],
    "start-after-centre-0.05": [[(1.1, 1.55), (1.1, 3.9)]],
    "start-after-centre-0.2": [[(1.1, 1.7), (1.1, 3.9)]],
    # Same probes reversed, and in x, to see if the asymmetry follows the line
    # direction (start vs end) or the geometry (high vs low coordinates).
    "reversed-end-before-centre-0.2": [[(1.1, 2.3), (1.1, 0.1)]],
    "reversed-start-after-centre-0.2": [[(1.1, 3.9), (1.1, 1.7)]],
    "end-x-before-centre-0.2": [[(0.1, 1.1), (2.3, 1.1)]],
    "reversed-end-x-before-centre-0.2": [[(2.3, 1.1), (0.1, 1.1)]],
    "start-x-after-centre-0.2": [[(1.7, 1.1), (3.9, 1.1)]],
    "reversed-start-x-after-centre-0.2": [[(3.9, 1.1), (1.7, 1.1)]],
    # Lines passing exactly through cell centres: ties for xugrid's centre rule.
    "tie-diagonal": [[(1.5, 1.5), (2.5, 2.5)]],
}

# name: (xmin, ymin, cellsize)
GRIDS = {
    "unit": (0.0, 0.0, 1.0),
    "rd": (100_000.0, 400_000.0, 100.0),
}

# Type aliases for clarity
# floats represent coordinates (x, y)
# integers represent cell indices for a raster
Coord = tuple[float, float]
Line = list[Coord]
Window = tuple[float, float, float, float]
Edge = tuple[tuple[int, int], tuple[int, int]]

# Cases where our result is known to differ from iMOD5 5.6.1: reported as xfail.
KNOWN_DIFFERENCES = {
    "diagonal-short": "Passes exactly through cell centres (a tie); see tie-diagonal.",
    "diagonal-full": "Passes exactly through cell centres (a tie); see tie-diagonal.",
    "tie-diagonal": (
        "Line through exact cell centres: iMOD5 tie behaviour is not understood "
        "and ours depends on floating point noise."
    ),
}


def write_ascii_gen(path: Path, lines: list[Line]) -> None:
    text = []
    for number, coordinates in enumerate(lines, start=1):
        text.append(str(number))
        text.extend(f"{x:.6f},{y:.6f}" for x, y in coordinates)
        text.append("END")
    text.append("END")
    path.write_text("\n".join(text) + "\n", encoding="ascii")


def run_imod5(
    case_dir: Path,
    gen_in: Path,
    gen_out: Path,
    window: Window,
    cellsize: float,
) -> subprocess.CompletedProcess:
    gen_out.unlink(missing_ok=True)
    inifile = case_dir / "snap.ini"
    inifile.write_text(
        "\n".join(
            [
                "FUNCTION=GENSNAPTOGRID",
                f"GENFILE_IN={gen_in}",
                f"GENFILE_OUT={gen_out}",
                f"WINDOW={','.join(f'{value:.3f}' for value in window)}",
                f"CELL_SIZE={cellsize}",
                "",
            ]
        ),
        encoding="ascii",
    )
    result = subprocess.run(
        [str(IMOD5_EXE), str(inifile)],
        cwd=case_dir,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=TIMEOUT_SECONDS,
        check=False
    )
    (case_dir / "imod5_console.txt").write_text(
        f"returncode: {result.returncode}\n"
        f"--- stdout\n{result.stdout}\n"
        f"--- stderr\n{result.stderr}\n",
        encoding="utf-8",
    )
    return result


def snap_with_ours(
    gen_in: Path,
    gen_out: Path,
    window: Window,
    cellsize: float,
) -> gpd.GeoDataFrame:
    gen_out.unlink(missing_ok=True)
    try:
        snap_gen_file_to_grid(gen_in, gen_out, window=window, cellsize=cellsize)
    except ValueError as e:
        if "No line features" not in str(e):
            raise
        return gpd.GeoDataFrame(geometry=[])
    return imod.gen.read(gen_out)


def read_gen_or_empty(path: Path) -> gpd.GeoDataFrame:
    """iMOD5 writes an empty result as a file containing only ``END``."""
    if not path.exists() or path.read_bytes().strip().upper() == b"END":
        return gpd.GeoDataFrame(geometry=[])
    return imod.gen.read(path)


def grid_edges(
    gdf: gpd.GeoDataFrame, xmin: float, ymin: float, cellsize: float
) -> set[Edge]:
    """
    Reduce lines to a set of unit grid edges, in grid node indices.

    Raises AssertionError on vertices off the grid nodes and on non-axis-aligned
    segments, since snapped output should have neither.
    """
    edges: set[Edge] = set()
    if len(gdf) == 0:
        return edges
    for part in shapely.get_parts(gdf.geometry.to_numpy()):
        if part.geom_type != "LineString":
            raise TypeError(f"Expected lines in output, got {part.geom_type}")
        index = (np.asarray(part.coords)[:, :2] - [xmin, ymin]) / cellsize
        rounded = np.round(index)
        assert np.abs(index - rounded).max() <= 0.01, (
            f"Vertices not on grid nodes (in cell units from origin):\n{index}"
        )
        nodes = rounded.astype(int)
        for start, end in zip(nodes[:-1], nodes[1:]):
            step = end - start
            assert step[0] == 0 or step[1] == 0, f"Diagonal segment {start} -> {end}"
            n = int(np.abs(step).max())
            unit = np.sign(step)
            for k in range(n):
                a = tuple(int(v) for v in start + k * unit)
                b = tuple(int(v) for v in start + (k + 1) * unit)
                edges.add(tuple(sorted((a, b))))  # type: ignore[arg-type]
    return edges


def save_comparison_plot(
    path: Path,
    lines: list[Line],
    ours: gpd.GeoDataFrame,
    theirs: gpd.GeoDataFrame,
    window: Window,
    cellsize: float,
) -> None:
    xmin, ymin, xmax, ymax = window
    fig = Figure(figsize=(7, 7))
    ax = fig.subplots()
    ax.set_xticks(np.arange(xmin, xmax + 0.5 * cellsize, cellsize))
    ax.set_yticks(np.arange(ymin, ymax + 0.5 * cellsize, cellsize))
    ax.tick_params(labelbottom=False, labelleft=False)
    ax.grid(color="0.8")
    for line in lines:
        x, y = zip(*line)
        ax.plot(x, y, color="green", linewidth=6, alpha=0.4, label="input")
    for part in shapely.get_parts(ours.geometry.to_numpy()):
        ax.plot(*part.xy, color="red", linewidth=3.5, label="ours")
    for part in shapely.get_parts(theirs.geometry.to_numpy()):
        ax.plot(*part.xy, color="blue", linewidth=1.5, linestyle="--", label="iMOD5")
    margin = 0.25 * cellsize
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_aspect("equal")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="upper left")
    fig.savefig(path, dpi=100)


def prepare_case(case: str, grid: str, work_dir: Path) -> tuple[Path, Path, list[Line], Window, float]:
    xmin, ymin, cellsize = GRIDS[grid]
    window = (xmin, ymin, xmin + 4 * cellsize, ymin + 4 * cellsize)
    case_dir = work_dir / f"{grid}_{case}"
    case_dir.mkdir(parents=True, exist_ok=True)
    lines: list[Line] = [
        [(xmin + x * cellsize, ymin + y * cellsize) for x, y in line]
        for line in CASES[case]
    ]
    gen_in = case_dir / "input.gen"
    write_ascii_gen(gen_in, lines)
    return case_dir, gen_in, lines, window, cellsize


@cache
def imod5_result(case: str, grid: str, work_dir: Path) -> tuple[gpd.GeoDataFrame, int, bool]:
    """Run iMOD5 once per case; returns output lines, return code, file exists."""
    case_dir, gen_in, _, window, cellsize = prepare_case(case, grid, work_dir)
    gen_out = case_dir / "imod5.gen"
    result = run_imod5(case_dir, gen_in, gen_out, window, cellsize)
    return read_gen_or_empty(gen_out), result.returncode, gen_out.exists()


@pytest.fixture(scope="session", autouse=True)
def imod5_works(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Fail once, with diagnostics, when iMOD5 cannot be driven from here."""
    case_dir = tmp_path_factory.mktemp("_snap_smoke")
    case_dir.mkdir(parents=True, exist_ok=True)
    gen_in = case_dir / "input.gen"
    gen_out = case_dir / "imod5.gen"
    write_ascii_gen(gen_in, CASES["horizontal"])
    result = run_imod5(case_dir, gen_in, gen_out, (0.0, 0.0, 4.0, 4.0), 1.0)
    if not gen_out.exists():
        pytest.fail(
            "iMOD5 did not write an output GEN for a simple horizontal line.\n"
            f"Command: {[str(IMOD5_EXE), str(case_dir / 'snap.ini')]}\n"
            f"Return code: {result.returncode}\n"
            f"stdout: {result.stdout[-1000:]}\nstderr: {result.stderr[-1000:]}\n"
            f"Run file:\n{(case_dir / 'snap.ini').read_text()}\n"
            "Try running that command by hand."
        )


@pytest.mark.parametrize("grid", list(GRIDS))
@pytest.mark.parametrize("case", list(CASES))
def test_snap_matches_imod5(tmp_path: Path, case: str, grid: str):
    xmin, ymin, cellsize = GRIDS[grid]
    case_dir, gen_in, lines, window, cellsize = prepare_case(case, grid, tmp_path)

    ours = snap_with_ours(gen_in, case_dir / "ours.gen", window, cellsize)
    theirs, returncode, wrote_output = imod5_result(case, grid, tmp_path)

    ours_edges = grid_edges(ours, xmin, ymin, cellsize)
    imod5_edges = grid_edges(theirs, xmin, ymin, cellsize)

    only_in_ours = sorted(ours_edges - imod5_edges)
    only_in_imod5 = sorted(imod5_edges - ours_edges)
    has_differences = bool(only_in_ours or only_in_imod5)
    
    report = (
        f"Edges differ (grid node indices; files in {case_dir}).\n"
        f"Only ours ({len(only_in_ours)}): {only_in_ours}\n"
        f"Only iMOD5 ({len(only_in_imod5)}): {only_in_imod5}\n"
        f"iMOD5 return code: {returncode}, wrote output: {wrote_output}"
    )
    (case_dir / "diff.txt").write_text(
        report if has_differences else "identical\n", encoding="utf-8"
    )
    save_comparison_plot(
        case_dir / "comparison.png", lines, ours, theirs, window, cellsize
    )

    if case in KNOWN_DIFFERENCES and has_differences:
        pytest.xfail(KNOWN_DIFFERENCES[case])
    assert not has_differences, report
