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


def _sorted_coordinates(gdf: gpd.GeoDataFrame) -> list[tuple]:
    """Return the vertices of each line in order, with the lines sorted."""
    return sorted(tuple(map(tuple, shapely.get_coordinates(line).tolist())) for line in gdf.geometry)


@pytest.fixture(scope="module")
def genpuzzle_out(tmp_path_factory: pytest.TempPathFactory) -> gpd.GeoDataFrame:
    genfile_out = tmp_path_factory.mktemp("genpuzzle") / "out.gen"
    genpuzzle(GENFILE_IN, genfile_out)
    return imod.formats.gen.read(genfile_out)


@pytest.fixture(scope="module")
def imod5_out() -> gpd.GeoDataFrame:
    return imod.formats.gen.read(IMOD5_OUT)


def test_output_matches_imod5(
    genpuzzle_out: gpd.GeoDataFrame, imod5_out: gpd.GeoDataFrame
) -> None:
    """Regression test: genpuzzle output matches the GENPUZZLE output of iMOD5."""
    assert _sorted_coordinates(genpuzzle_out) == _sorted_coordinates(imod5_out)


def test_tolerance_controls_snapping(tmp_path: Path) -> None:
    small = genpuzzle(GENFILE_IN, tmp_path / "small.gen", tolerance=0.0001)
    large = genpuzzle(GENFILE_IN, tmp_path / "large.gen", tolerance=3.0)

    assert len(small) == 15
    assert len(large) == 13


def test_z_coordinates_are_forced_to_2d(tmp_path: Path) -> None:
    """Input with Z values gives the same 2D output as the input without Z."""
    segments = imod.formats.gen.read(GENFILE_IN)
    # Give each segment its own Z, so connected segments disagree at shared nodes.
    lines = []
    for i, geom in enumerate(segments.geometry):
        lines.append(f"segment_{i}")
        lines.extend(f"{x} {y} {i * 0.5}" for x, y in shapely.get_coordinates(geom))
        lines.append("end")
    lines.append("end")
    genfile_z = tmp_path / "z.gen"
    genfile_z.write_text("\n".join(lines) + "\n")

    out_2d = genpuzzle(GENFILE_IN, tmp_path / "out_2d.gen")
    out_z = genpuzzle(genfile_z, tmp_path / "out_z.gen")

    assert not out_z.has_z.any()
    assert _sorted_coordinates(out_z) == _sorted_coordinates(out_2d)


PLOT_TEST_CASES = [
    ("Exact joins", (30, 5), (0, 5)),
    ("Larger gap", (21.5, 25), (0, 30)),
    ("Converging line", (20, 50), (0, 56)),
    ("Isolated line", (70, 20), (53, 43)),
    ("Diverging lines", (40, 70), (0, 70)),
    ("Three-way junction", (60, 65), (43, 84)),
    ("0.00099 gap", (20, 90), (0, 97)),
    ("Square ring", (60, 95), (55, 103)),
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
            coords = shapely.get_coordinates(line)
            (x0, y0), (x1, y1) = coords[0], coords[-1]
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
