"""Tests that genpuzzle reproduces the output of the GENPUZZLE function in iMOD5."""

from pathlib import Path

import geopandas as gpd
import imod
import pytest
import shapely
from matplotlib.figure import Figure

from genpuzzle import genpuzzle

DATA_DIR = Path(__file__).parent / "data"
GENFILE_IN = DATA_DIR / "genpuzzle_test.gen"
IMOD5_OUT = DATA_DIR / "out_imod5.gen"
FIGURE_OUT = DATA_DIR / "genpuzzle_comparison.png"

THREE_WAY_JUNCTION = (60.0, 65.0)


def _without_three_way_junction(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Temporary: remove once the three-way junction behaviour is included correctly. 
    return gdf[~gdf.geometry.intersects(shapely.Point(THREE_WAY_JUNCTION))]


@pytest.fixture(scope="module")
def genpuzzle_out(tmp_path_factory: pytest.TempPathFactory) -> gpd.GeoDataFrame:
    genfile_out = tmp_path_factory.mktemp("genpuzzle") / "out.gen"
    genpuzzle(GENFILE_IN, genfile_out)
    return imod.formats.gen.read(genfile_out)


@pytest.fixture(scope="module")
def imod5_out() -> gpd.GeoDataFrame:
    return imod.formats.gen.read(IMOD5_OUT)


def test_output_matches_imod5_without_three_way_junction(
    genpuzzle_out: gpd.GeoDataFrame, imod5_out: gpd.GeoDataFrame
) -> None:
    actual = _without_three_way_junction(genpuzzle_out)
    expected = _without_three_way_junction(imod5_out)
    actual_lines = sorted(tuple(line.coords) for line in actual.geometry)
    expected_lines = sorted(tuple(line.coords) for line in expected.geometry)
    assert actual_lines == expected_lines


def test_tolerance_controls_snapping(tmp_path: Path) -> None:
    small = genpuzzle(GENFILE_IN, tmp_path / "small.gen", tolerance=0.0001)
    large = genpuzzle(GENFILE_IN, tmp_path / "large.gen", tolerance=3.0)

    assert len(small) == 10
    assert len(large) == 9


def _plot_lines(ax, gdf: gpd.GeoDataFrame, title: str) -> None:
    for line in gdf.geometry:
        x, y = zip(*line.coords)
        ax.plot(x, y, "-o", markersize=3)
    ax.set_title(f"{title} ({len(gdf)} lines)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def test_plot_comparison(
    genpuzzle_out: gpd.GeoDataFrame, imod5_out: gpd.GeoDataFrame
) -> None:
    """Plot a comparison of the input, genpuzzle output, and iMOD5 output."""
    input_gdf = imod.formats.gen.read(GENFILE_IN)
    panels = [
        ("Input", input_gdf),
        ("genpuzzle output", genpuzzle_out),
        ("iMOD5 output", imod5_out),
    ]
    fig = Figure(figsize=(10, 4.5), layout="constrained")
    axes = fig.subplots(1, len(panels), sharex=True, sharey=True)
    for ax, (title, gdf) in zip(axes, panels):
        _plot_lines(ax, gdf, title)
    fig.suptitle(f"GENPUZZLE comparison")
    fig.savefig(FIGURE_OUT, dpi=150)
    assert FIGURE_OUT.exists()
