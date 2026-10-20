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


PLOT_TEST_CASES = [
    ("Exact joins", (30, 5), (0, 5)),
    ("Larger gap", (21.5, 25), (0, 30)),
    ("Converging line", (20, 50), (0, 56)),
    ("Isolated line", (70, 20), (53, 43)),
    ("Floating point precision", (20, 75), (0, 83)),
    ("Three-way junction", (60, 65), (43, 87)),
    ("0.0001 gap", (20, 90), (0, 97)),
]


def test_plot_comparison(
    genpuzzle_out: gpd.GeoDataFrame, imod5_out: gpd.GeoDataFrame
) -> None:
    """Plot a comparison of the input, genpuzzle output, and iMOD5 output."""
    panels = [
        (GENFILE_IN.name, imod.formats.gen.read(GENFILE_IN)),
        ("genpuzzle output", genpuzzle_out),
        (IMOD5_OUT.name, imod5_out),
    ]
    fig = Figure(figsize=(3.3 * len(panels), 5))
    axes = fig.subplots(1, len(panels), sharex=True, sharey=True)
    for ax, (name, gdf) in zip(axes, panels):
        labels = gdf.get("id", gdf.index.to_series())
        gdf.plot(ax=ax, column=labels, cmap="tab10", linewidth=2)
        for line in gdf.geometry:
            (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
            ax.plot([x0, x1], [y0, y1], "ko", markersize=4)
        ax.set_title(f"{name} ({len(gdf)} lines)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    for label, point, text_position in PLOT_TEST_CASES:
        axes[0].annotate(label, xy=point, xytext=text_position, fontsize=8)
    axes[0].set_ylim(-15, 110)

    fig.tight_layout(pad=0.5, w_pad=0.5)
    fig.savefig(FIGURE_OUT, dpi=150, bbox_inches="tight")
    assert FIGURE_OUT.exists()
