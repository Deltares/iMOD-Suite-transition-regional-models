"""
Unit tests of snapping: the grid helpers, wall assembly and the public functions.
Accepted differences from iMOD5 itself are tested in
``test_snapping_imod5.py``.
"""

import logging
from pathlib import Path

import geopandas as gpd
import imod
import numpy as np
import pandas as pd
import pytest
import shapely
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from imod.formats.gen import read_ascii
from imod.util import empty_2d

from src.cli import _gensnaptogrid
from src.grid_helpers import GridSpec, create_target_grid, grid_spec, open_idf_grid
from src.grid_snapping import (
    raster_order,
    snap_lines_to_edges,
    unique_walls,
    vertical_walls,
    wall_end_points,
)
from src.snapping import snap_gen_file_to_grid, snap_gen_to_grid

HORIZONTAL = shapely.LineString([(0.1, 1.1), (3.9, 1.1)])
VERTICAL = shapely.LineString([(1.1, 0.1), (1.1, 3.9)])
CORNER = shapely.LineString([(0.6, 1.4), (2.6, 1.4), (2.6, 3.4)])


def _grid(xmin=0.0, xmax=4.0, ymin=0.0, ymax=4.0, cellsize=1.0) -> xr.DataArray:
    return empty_2d(dx=cellsize, xmin=xmin, xmax=xmax, dy=-cellsize, ymin=ymin, ymax=ymax)


def _nonuniform_grid(shift: float = 0.0) -> xr.DataArray:
    # Cell sizes 1, 2 and 3 along x: the cell centres are not evenly spaced.
    return xr.DataArray(
        np.ones((3, 3)),
        dims=("y", "x"),
        coords={
            "x": [0.5 + shift, 2.0 + shift, 4.5 + shift],
            "y": [3.5, 2.0, 0.5],
            "dx": ("x", [1.0, 2.0, 3.0]),
            "dy": ("y", [-1.0, -2.0, -1.0]),
        },
    )


@pytest.fixture
def like() -> xr.DataArray:
    return _grid()


# Grid helpers


def test_target_grid_from_like_window_and_cellsize(like):
    xr.testing.assert_identical(create_target_grid(like, None, None), like)
    xr.testing.assert_identical(
        create_target_grid(None, (0.0, 0.0, 4.0, 4.0), 1.0), like
    )
    actual = create_target_grid(like, (1.0, 1.0, 3.0, 3.0), None)
    xr.testing.assert_identical(actual, like.sel(x=slice(1.0, 3.0), y=slice(3.0, 1.0)))
    off_centre = create_target_grid(like, (0.6, 0.6, 3.4, 3.4), None)
    np.testing.assert_array_equal(off_centre.x, [1.5, 2.5])
    np.testing.assert_array_equal(off_centre.y, [2.5, 1.5])


@pytest.mark.parametrize(
    "window, cellsize, x_edges, y_edges",
    [
        ((0.0, 0.0, 4.6, 4.6), 1.0, np.arange(5.0), np.arange(5.0)),
        (
            (100000.0, 400000.0, 100330.0, 400280.0),
            100.0,
            100000.0 + 100.0 * np.arange(4),
            400000.0 + 100.0 * np.arange(3),
        ),
    ],
)
def test_target_grid_truncates_window_like_imod5(window, cellsize, x_edges, y_edges):
    spec = grid_spec(create_target_grid(None, window, cellsize))
    assert (spec.ncol, spec.nrow) == (len(x_edges) - 1, len(y_edges) - 1)
    assert (spec.xmin, spec.ymax) == (x_edges[0], y_edges[-1])


@pytest.mark.parametrize(
    "use_like, window, cellsize, match",
    [
        (False, None, None, "window and cellsize"),
        (False, None, 1.0, "window and cellsize"),
        (False, (0.0, 0.0, 4.0, 4.0), None, "window and cellsize"),
        (False, (0.0, 0.0, 4.0, 4.0), 0.0, "cellsize"),
        (False, (0.0, 0.0, 4.0, 4.0), np.nan, "cellsize"),
        (False, (0.0, 0.0, 0.9, 4.0), 1.0, "window"),
        (True, (4.0, 0.0, 0.0, 4.0), 1.0, "window"),
        (True, (0.0, 0.0, 4.0, np.inf), 1.0, "window"),
        (True, None, 2.0, "cellsize"),
        (True, (10.0, 10.0, 12.0, 12.0), None, "window"),
    ],
)
def test_target_grid_rejects_invalid_arguments(like, use_like, window, cellsize, match):
    with pytest.raises(ValueError, match=match):
        create_target_grid(like if use_like else None, window, cellsize)


def test_target_grid_requires_two_dimensions(like):
    with pytest.raises(ValueError, match="two-dimensional"):
        create_target_grid(like.expand_dims(layer=[1]), None, None)


def test_grid_spec(like):
    grid = empty_2d(dx=2.0, xmin=0.0, xmax=6.0, dy=-1.0, ymin=0.0, ymax=4.0)
    expected = GridSpec(0.0, 4.0, 2.0, 1.0, 3, 4)
    assert grid_spec(grid) == expected
    assert grid_spec(grid.isel(y=slice(None, None, -1))) == expected

    single = empty_2d(dx=2.0, xmin=0.0, xmax=2.0, dy=-1.0, ymin=0.0, ymax=1.0)
    with pytest.raises(ValueError):
        grid_spec(single.drop_vars(["dx", "dy"], errors="ignore"))
    assert grid_spec(single) == GridSpec(0.0, 1.0, 2.0, 1.0, 1, 1)

    spec = grid_spec(_nonuniform_grid())
    np.testing.assert_allclose(spec.xnodes, [0.0, 1.0, 3.0, 6.0])
    np.testing.assert_allclose(spec.ynodes, [4.0, 3.0, 1.0, 0.0])


def test_grid_spec_of_an_idf_reads_the_header_as_imod5_does(tmp_path):
    # iMOD5 reads the real values of the IDF header as single precision text with
    # about 7 significant digits: ymax 462821.75 becomes 462821.8.
    grid = empty_2d(
        dx=6.25, xmin=150000.0, xmax=150018.75, dy=-6.25, ymin=462753.0, ymax=462821.75
    )
    imod.idf.save(tmp_path / "grid", grid.astype(np.float32))
    from_idf = open_idf_grid(next(tmp_path.glob("grid*.idf")))

    assert grid_spec(from_idf).ymax == 462821.8
    # Cells selected from the grid are no longer the grid of the header.
    assert grid_spec(from_idf.isel(y=slice(1, None))).ymax == 462815.5
    assert grid_spec(grid).ymax == 462821.75


# Walls are (kind, irow, icol): kind 0 is the vertical wall on the
# east edge of cell (irow, icol), kind 1 the horizontal wall on its south edge.

GRID = GridSpec(0.0, 4.0, 1.0, 1.0, 4, 4)

def _walls(lines, grid=GRID, snap=snap_lines_to_edges) -> set[tuple[int, int, int]]:
    result = snap([np.asarray(line, dtype=float) for line in lines], grid)
    return set(zip(result["kind"], result["irow"], result["icol"]))


@pytest.mark.parametrize(
    "lines, expected",
    [
        ([[(0.1, 1.1), (3.9, 1.1)]], {(1, 2, c) for c in range(4)}),
        ([[(1.1, 0.1), (1.1, 3.9)]], {(0, r, 0) for r in range(4)}),
        (
            [[(0.6, 1.4), (2.6, 1.4), (2.6, 3.4)]],
            {(1, 2, 1), (1, 2, 2), (0, 1, 2), (0, 2, 2)},
        ),
        ([[(1.5, 1.5), (2.5, 2.5)]], {(0, 1, 1), (0, 2, 0), (1, 1, 1), (1, 0, 2)}),
        ([[(1.1, 0.1), (1.1, 2.3)]], {(0, 2, 0), (0, 3, 0)}),
        ([[(10.0, 10.0), (12.0, 12.0)]], set()),
        ([[(2.0, 2.0)]], set()),
        ([[]], set()),
        ([[(1.1, 1.1), (1.1, 1.1), (2.9, 2.9)]], {(0, 1, 1), (0, 2, 0), (1, 0, 2), (1, 1, 1)}),
    ],
)
def test_snap_lines_to_edges_walls(lines, expected):
    assert _walls(lines) == expected


def test_snap_lines_to_edges_preserves_feature_positions():
    result = snap_lines_to_edges(
        [np.array([(1.1, 0.1), (1.1, 2.3)]), np.array([(2.1, 0.1), (2.1, 3.9)])], GRID
    )
    first = result[result["position"] == 0]
    assert set(zip(first["kind"], first["irow"], first["icol"])) == {(0, 2, 0), (0, 3, 0)}
    assert list(result.columns) == ["position", "kind", "irow", "icol"]
    assert snap_lines_to_edges([], GRID).empty


def test_snap_lines_to_edges_on_a_nonuniform_grid():
    nonuniform = GridSpec(
        0.0,
        4.0,
        float("nan"),
        float("nan"),
        4,
        4,
        xnodes=np.arange(5.0),
        ynodes=np.arange(4.0, -1.0, -1.0),
    )
    actual = _walls([[(0.1, 1.1), (3.9, 1.1)], [(1.1, 0.1), (1.1, 3.9)]], nonuniform)
    assert {(1, 2, 0), (1, 2, 1), (1, 2, 2), (0, 0, 0), (0, 1, 0), (0, 2, 0)} <= actual


def test_snap_lines_to_edges_keeps_interior_walls_sorted_by_position():
    lines = [
        np.array([(10.0, 10.0), (12.0, 12.0)]),
        np.array([(0.1, 1.1), (3.9, 1.1)]),
        np.array([(0.1, 1.1), (3.9, 1.1)]),
    ]
    result = snap_lines_to_edges(lines, GRID)
    assert result["position"].tolist() == [1] * 4 + [2] * 4
    np.testing.assert_allclose(wall_end_points(GRID, result)[:, :, 1], 1.0)

    assert snap_lines_to_edges([np.array([(0.1, 0.0), (3.9, 0.0)])], GRID).empty
    assert snap_lines_to_edges([np.array([(1.1, 1.1), (1.2, 1.1)])], GRID).empty


def test_wall_end_points():
    edges = pd.DataFrame({"position": [0, 0], "kind": [0, 1], "irow": [1, 1], "icol": [0, 0]})
    vertical, horizontal = wall_end_points(GridSpec(0.0, 4.0, 2.0, 1.0, 2, 4), edges)
    # East edge of the cell (1, 0), north to south; its south edge, west to east.
    np.testing.assert_array_equal(vertical, [(2.0, 3.0), (2.0, 2.0)])
    np.testing.assert_array_equal(horizontal, [(0.0, 2.0), (2.0, 2.0)])

    nonuniform = GridSpec(
        0.0,
        4.0,
        float("nan"),
        float("nan"),
        3,
        3,
        xnodes=np.array([0.0, 1.0, 3.0, 4.0]),
        ynodes=np.array([4.0, 3.0, 1.0, 0.0]),
    )
    edges = pd.DataFrame({"position": [0, 0], "kind": [0, 1], "irow": [1, 0], "icol": [1, 2]})
    vertical, horizontal = wall_end_points(nonuniform, edges)
    np.testing.assert_array_equal(vertical, [(3.0, 3.0), (3.0, 1.0)])
    np.testing.assert_array_equal(horizontal, [(3.0, 3.0), (4.0, 3.0)])


def test_raster_order_and_unique_walls():
    edges = pd.DataFrame(
        {
            "position": [0, 0, 1, 1],
            "kind": [0, 1, 1, 0],
            "irow": [1, 0, 0, 0],
            "icol": [0, 0, 0, 5],
        }
    )
    assert raster_order(edges).tolist() == [1, 2, 3, 0]
    # A wall snapped by two features is kept once, with the first feature.
    unique = unique_walls(edges.iloc[[0, 1, 2, 3, 1]].reset_index(drop=True))
    assert list(zip(unique["position"], unique["kind"], unique["irow"], unique["icol"])) == [
        (0, 1, 0, 0),
        (1, 0, 0, 5),
        (0, 0, 1, 0),
    ]


def test_vertical_walls_use_both_cells_and_drop_nodata():
    edges = pd.DataFrame(
        {"position": [0, 1, 2], "kind": [0, 1, 0], "irow": [0, 0, 1], "icol": [0, 1, 1]}
    )
    top = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, np.nan], [70.0, 80.0, 90.0]])

    kept, polygons = vertical_walls(GRID, edges, top, top - 5.0)

    # The third edge touches the nodata cell (1, 2).
    assert kept["position"].tolist() == [0, 1]
    # East of cell (0, 0): north to south, max of the TOPs, min of the BOTs.
    assert list(polygons[0].exterior.coords) == [
        (1.0, 4.0, 20.0),
        (1.0, 3.0, 20.0),
        (1.0, 3.0, 5.0),
        (1.0, 4.0, 5.0),
        (1.0, 4.0, 20.0),
    ]
    # South of cell (0, 1): west to east.
    assert list(polygons[1].exterior.coords)[:2] == [(1.0, 3.0, 50.0), (2.0, 3.0, 50.0)]


def test_vertical_walls_keep_zero_and_negative_height_and_may_be_empty():
    edges = pd.DataFrame({"position": [0], "kind": [0], "irow": [1], "icol": [1]})
    top = np.full((4, 4), 1000.0)
    for bottom in (top, top + 5.0):
        _, polygons = vertical_walls(GRID, edges, top, bottom)
        assert [z for *_, z in polygons[0].exterior.coords][2] == bottom[0, 0]

    nodata = np.full((4, 4), np.nan)
    kept, polygons = vertical_walls(GRID, edges, nodata, nodata)
    assert kept.empty and len(polygons) == 0


# snap_gen_to_grid, 2D


@pytest.mark.parametrize(
    "coordinates",
    [
        [(0.1, 1.1), (3.9, 1.1)],
        [(1.1, 1.1), (2.9, 2.9)],
        [(-1.0, 1.1), (5.0, 1.1)],
        [(1.0, 1.0), (3.0, 3.0)],
    ],
    ids=["horizontal", "diagonal", "crossing", "through-nodes"],
)
def test_snap_gen_to_grid_gives_unit_edges_and_keeps_attributes(like, coordinates):
    original = gpd.GeoDataFrame(
        {"id": ["line"], "value": [3.0]},
        geometry=[shapely.LineString(coordinates)],
        crs="EPSG:28992",
    )
    actual = snap_gen_to_grid(original, like=like)

    assert actual.crs == original.crs
    assert actual[["id", "value"]].to_dict("list") == {"id": ["line"], "value": [3.0]}
    line = actual.geometry.iloc[0]
    assert line.geom_type == "LineString"
    xy = np.asarray(line.coords)
    np.testing.assert_allclose(xy, np.round(xy))
    np.testing.assert_allclose(np.abs(np.diff(xy, axis=0)).sum(axis=1), 1.0)
    assert ((xy >= 0.0) & (xy <= 4.0)).all()


def test_snap_gen_to_grid_keeps_features_apart_and_does_not_modify_input(like):
    original = gpd.GeoDataFrame(
        {"id": ["outside", "first", "second"]},
        index=[7, 7, 12],
        geometry=[shapely.LineString([(10, 10), (12, 12)]), HORIZONTAL, HORIZONTAL],
        crs="EPSG:28992",
    )
    unchanged = original.copy(deep=True)

    actual = snap_gen_to_grid(original, like=like)

    assert actual["id"].tolist() == ["first", "second"]
    assert actual.index.tolist() == [0, 1]
    assert actual.geometry.iloc[0].equals(actual.geometry.iloc[1])
    assert_geodataframe_equal(original, unchanged)


def test_snap_gen_to_grid_keeps_disconnected_parts_in_one_row(like):
    original = gpd.GeoDataFrame(
        {"id": ["reentering"]},
        geometry=[shapely.LineString([(1, 1), (5, 1), (5, 3), (1, 3)])],
    )
    actual = snap_gen_to_grid(original, like=like)
    assert actual["id"].tolist() == ["reentering"]
    assert len(actual.geometry.iloc[0].geoms) == 2


@pytest.mark.parametrize(
    "geometries",
    [
        [],
        [shapely.LineString([(10, 10), (12, 12)])],
        [shapely.LineString([(1.01, 1.01), (1.02, 1.02)])],
        [shapely.LineString([(0.0, 0.0), (4.0, 0.0)])],
        [shapely.LineString([(4.0, 4.0), (0.0, 4.0)])],
        [shapely.LineString([(0.0, 4.0), (0.0, 0.0)])],
    ],
    ids=["no-features", "outside", "short", "bottom", "top", "left"],
)
def test_snap_gen_to_grid_nothing_snaps(like, geometries):
    original = gpd.GeoDataFrame(
        {"id": [str(i) for i in range(len(geometries))]},
        geometry=geometries,
        crs="EPSG:28992",
    )
    actual = snap_gen_to_grid(original, like=like)
    assert actual.empty
    assert actual.columns.tolist() == original.columns.tolist()
    assert actual.crs == original.crs


def test_snap_gen_to_grid_single_cell_has_only_boundary_edges():
    original = gpd.GeoDataFrame(geometry=[shapely.LineString([(0.1, 0.1), (0.9, 0.9)])])
    assert snap_gen_to_grid(original, window=(0.0, 0.0, 1.0, 1.0), cellsize=1.0).empty


@pytest.mark.parametrize("distance", [-1.0, np.nan, np.inf])
def test_snap_gen_to_grid_rejects_invalid_distance(like, distance):
    original = gpd.GeoDataFrame(geometry=[shapely.LineString([(1, 1), (3, 3)])])
    with pytest.raises(ValueError, match="max_snap_distance"):
        snap_gen_to_grid(original, like=like, max_snap_distance=distance)


@pytest.mark.parametrize("geometry", [shapely.Point(1, 1), shapely.LineString(), None])
def test_snap_gen_to_grid_rejects_invalid_geometry(like, geometry):
    with pytest.raises(ValueError, match="non-empty LineString"):
        snap_gen_to_grid(gpd.GeoDataFrame(geometry=[geometry]), like=like)


def test_snap_gen_to_grid_ignores_z_and_snaps_the_ring_of_a_polygon(like):
    xy = [(0.6, 1.4), (2.6, 1.4), (2.6, 3.4), (0.6, 3.4), (0.6, 1.4)]
    expected = snap_gen_to_grid(
        gpd.GeoDataFrame({"id": ["a"]}, geometry=[shapely.LineString(xy)]), like=like
    )
    with_z = gpd.GeoDataFrame(
        {"id": ["a"]}, geometry=[shapely.LineString([(x, y, 5.0) for x, y in xy])]
    )
    polygon = gpd.GeoDataFrame({"id": ["a"]}, geometry=[shapely.Polygon(xy)])

    assert not expected.empty
    assert_geodataframe_equal(snap_gen_to_grid(with_z, like=like), expected)
    assert_geodataframe_equal(snap_gen_to_grid(polygon, like=like), expected)


def test_snap_gen_to_grid_does_not_extend_the_last_feature(like):
    def length(*lines):
        gdf = gpd.GeoDataFrame(geometry=[shapely.LineString(line) for line in lines])
        return snap_gen_to_grid(gdf, like=like).geometry.length.sum()

    assert length([(1.1, 0.1), (1.1, 2.3)]) == 2.0
    assert length([(1.1, 2.3), (1.1, 0.1)]) == 2.0
    assert length([(1.1, 0.1), (1.1, 2.3)], [(2.1, 0.1), (2.1, 3.9)]) == 3.0 + 4.0 - 1.0


def test_snap_gen_to_grid_uses_xugrid_diagonal_tie_selection(like):
    original = gpd.GeoDataFrame(geometry=[shapely.LineString([(1.5, 1.5), (2.5, 2.5)])])
    assert snap_gen_to_grid(original, like=like).geometry.iloc[0].length == 4.0


def test_snap_gen_to_grid_max_snap_distance_moves_vertices_onto_nodes(like):
    def snap(coordinates, **kwargs):
        return snap_gen_to_grid(
            gpd.GeoDataFrame(geometry=[shapely.LineString(coordinates)]), like=like, **kwargs
        )

    line = [(0.9, 1.1), (3.1, 2.9)]
    moved = snap(line, max_snap_distance=0.2)
    assert_geodataframe_equal(moved, snap([(1.0, 1.0), (3.0, 3.0)]))
    assert not moved.geometry.iloc[0].equals(snap(line).geometry.iloc[0])
    assert snap([(0.95, 1.0), (1.05, 1.0)], max_snap_distance=0.1).empty


def test_snap_gen_to_grid_keeps_the_geometry_column_name(like):
    original = gpd.GeoDataFrame(
        {"id": ["line"]}, geometry=[shapely.LineString([(1.1, 1.1), (2.9, 2.9)])]
    ).rename_geometry("line_geometry")
    assert snap_gen_to_grid(original, like=like).geometry.name == "line_geometry"


@pytest.mark.parametrize("dimension", ["x", "y"])
def test_snap_gen_to_grid_on_a_grid_with_one_row_or_column(like, dimension):
    coordinates = [(0.1, 1.1), (3.9, 1.1)] if dimension == "x" else [(1.1, 0.1), (1.1, 3.9)]
    original = gpd.GeoDataFrame(geometry=[shapely.LineString(coordinates)])
    actual = snap_gen_to_grid(original, like=like.isel({dimension: [1]}))
    assert actual.geometry.length.sum() == 1.0


def test_snap_gen_to_grid_on_a_nonuniform_grid():
    original = gpd.GeoDataFrame(geometry=[shapely.LineString([(0.1, 1.1), (5.9, 1.1)])])
    actual = snap_gen_to_grid(original, like=_nonuniform_grid())
    xy = shapely.get_coordinates(actual.geometry.iloc[0])
    assert set(np.round(xy[:, 0], 6)) <= {0.0, 1.0, 3.0, 6.0}
    assert set(np.round(xy[:, 1], 6)) <= {0.0, 1.0, 3.0, 4.0}


# snap_gen_to_grid with top and bottom: vertical walls


def _coded(grid: xr.DataArray, sign: float = 1.0) -> xr.DataArray:
    """``1000 + 10 * col + row``, or ``1100 - 10 * col - row`` with ``sign=-1``."""
    col, row = np.floor(grid["x"]), np.floor(grid["y"])
    top = 1000.0 + 10.0 * col + row if sign > 0 else 1100.0 - 10.0 * col - row
    return top.transpose("y", "x")


def _cell(da: xr.DataArray, col: int, row: int) -> xr.DataArray:
    inside = (da["x"] >= col) & (da["x"] < col + 1) & (da["y"] >= row) & (da["y"] < row + 1)
    return inside.transpose("y", "x")


def _rings(gdf: gpd.GeoDataFrame) -> list[tuple[tuple[float, float, float], ...]]:
    return [tuple(polygon.exterior.coords) for polygon in gdf.geometry]


def _wall(a, b, ztop, zbot):
    """Ring as iMOD5 writes it: A-top, B-top, B-bot, A-bot, A-top."""
    (ax, ay), (bx, by) = a, b
    return ((ax, ay, ztop), (bx, by, ztop), (bx, by, zbot), (ax, ay, zbot), (ax, ay, ztop))


def _snap3d(lines, top, bottom=None, like=None, **kwargs):
    gdf = gpd.GeoDataFrame(geometry=list(lines))
    bottom = top - 500.0 if bottom is None else bottom
    return snap_gen_to_grid(gdf, like=like, top=top, bottom=bottom, **kwargs)


def test_walls_have_the_ring_order_and_direction_of_imod5(like):
    top = _coded(like)

    # In raster order: the walls of cell (1, 2) before those of cell (2, 2).
    assert _rings(_snap3d([CORNER], top, like=like)) == [
        _wall((3, 3), (3, 2), 1032, 522),
        _wall((1, 1), (2, 1), 1011, 510),
        _wall((3, 2), (3, 1), 1031, 521),
        _wall((2, 1), (3, 1), 1021, 520),
    ]
    assert _wall((1, 4), (1, 3), 1013, 503) in _rings(_snap3d([VERTICAL], top, like=like))


def test_centre_line_3d_has_parallel_walls_with_adjacent_cell_values(like):
    centre = shapely.LineString([(0.5, 0), (0.5, 4)])
    assert _rings(_snap3d([centre], _coded(like), like=like)) == [
        _wall((1, y + 1), (1, y), 1010 + y, 500 + y)
        for y in range(3, -1, -1)
    ]


def test_walls_take_the_highest_top_and_lowest_bottom_of_the_two_cells(like):
    # With values decreasing in x and y, the higher TOP is on the left or lower
    # side of an edge, and the lower BOT on the right or upper side.
    actual = _snap3d([VERTICAL, HORIZONTAL], _coded(like, sign=-1.0), like=like)

    rings = _rings(actual)
    assert _wall((1, 4), (1, 3), 1097, 587) in rings
    assert _wall((0, 1), (1, 1), 1100, 599) in rings


def test_walls_are_written_once_with_the_attributes_of_the_first_feature(like):
    other = shapely.LineString([(0.2, 1.2), (3.8, 1.2)])
    gdf = gpd.GeoDataFrame({"id": ["a", "b"]}, geometry=[HORIZONTAL, other])

    actual = snap_gen_to_grid(gdf, like=like, top=_coded(like), bottom=_coded(like) - 500.0)

    assert actual["id"].tolist() == ["a"] * 4


@pytest.mark.parametrize(
    "line, cell, nodata_in",
    [
        (VERTICAL, (1, 1), "both"),
        (VERTICAL, (1, 1), "top"),
        (VERTICAL, (1, 1), "bottom"),
        (VERTICAL, (0, 1), "both"),
        (HORIZONTAL, (1, 1), "both"),
        (HORIZONTAL, (1, 0), "both"),
    ],
)
def test_walls_next_to_a_nodata_cell_are_dropped(like, line, cell, nodata_in):
    top = _coded(like)
    bottom = top - 500.0
    masked = ~_cell(top, *cell)
    if nodata_in in ("both", "top"):
        top = top.where(masked)
    if nodata_in in ("both", "bottom"):
        bottom = bottom.where(masked)

    actual = _snap3d([line], top, bottom, like)

    assert len(actual) == 3
    assert not np.isnan(np.concatenate([p.exterior.coords for p in actual.geometry])).any()


def test_walls_on_window_and_larger_or_nonuniform_grids(like):
    big = _coded(_grid(xmin=-1.0, xmax=6.0, ymin=-2.0, ymax=5.0))
    # TOP and BOT may cover more than the grid.
    actual = _snap3d([HORIZONTAL], big, window=(0.0, 0.0, 4.0, 4.0), cellsize=1.0)
    assert _wall((0, 1), (1, 1), 1001, 500) in _rings(actual) and len(actual) == 4
    # Only the cells in the window are sampled.
    actual = _snap3d([HORIZONTAL], _coded(like), like=like, window=(1.0, 0.0, 3.0, 4.0))
    assert _rings(actual) == [_wall((1, 1), (2, 1), 1011, 510), _wall((2, 1), (3, 1), 1021, 520)]

    nonuniform = _nonuniform_grid()
    line = shapely.LineString([(0.1, 1.1), (5.9, 1.1)])
    actual = _snap3d([line], nonuniform * 1000.0, like=nonuniform)
    assert not actual.empty and (actual.geometry.geom_type == "Polygon").all()


@pytest.mark.parametrize("misaligned", [_grid(xmin=0.5, xmax=4.5), _grid(cellsize=0.5)])
def test_walls_reject_top_and_bottom_on_another_grid(like, misaligned):
    top = _coded(like)
    with pytest.raises(ValueError, match="grid"):
        _snap3d([HORIZONTAL], _coded(misaligned), top, like)
    with pytest.raises(ValueError, match="grid"):
        _snap3d([HORIZONTAL], top, _coded(misaligned), like)
    with pytest.raises(ValueError, match="two-dimensional"):
        _snap3d([HORIZONTAL], top.expand_dims(layer=[1]), top, like)

    # The spacing of a nonuniform grid is not known to be equal, so the alignment
    # has to follow from the cell centres alone.
    nonuniform = _nonuniform_grid()
    line = shapely.LineString([(0.1, 1.1), (5.9, 1.1)])
    with pytest.raises(ValueError, match="grid"):
        _snap3d([line], _nonuniform_grid(shift=0.9), nonuniform, nonuniform)


def test_walls_nothing_snapped_returns_an_empty_frame(like):
    gdf = gpd.GeoDataFrame({"id": ["s"]}, geometry=[shapely.LineString([(1.1, 1.1), (1.2, 1.1)])])
    actual = snap_gen_to_grid(gdf, like=like, top=_coded(like), bottom=_coded(like))
    assert actual.empty and actual.columns.tolist() == gdf.columns.tolist()


def test_top_and_bottom_must_be_given_together(like):
    gdf = gpd.GeoDataFrame(geometry=[HORIZONTAL])
    for kwargs in ({"top": like}, {"bottom": like}):
        with pytest.raises(ValueError, match="together"):
            snap_gen_to_grid(gdf, like=like, **kwargs)


# Files and the command line interface


@pytest.fixture
def gen_in(tmp_path) -> Path:
    path = tmp_path / "in.gen"
    path.write_text("1\n0.1,1.1\n3.9,1.1\nEND\nEND\n", encoding="ascii")
    return path


@pytest.fixture
def idfs(tmp_path, like) -> tuple[Path, Path]:
    top = _coded(like).astype(np.float32)
    imod.idf.save(tmp_path / "top", top)
    imod.idf.save(tmp_path / "bot", top - 500.0)
    return next(tmp_path.glob("top*.idf")), next(tmp_path.glob("bot*.idf"))


@pytest.fixture
def restore_log_handlers():
    imod_logger = logging.getLogger("imod")
    before = list(imod_logger.handlers)
    yield
    for handler in list(imod_logger.handlers):
        if handler not in before:
            handler.close()
            imod_logger.removeHandler(handler)


def test_file_without_idfs_is_a_binary_gen_with_merged_lines_and_attributes(tmp_path, like):
    original = gpd.GeoDataFrame(
        {"id": ["first", "reentering"], "name": ["a", "b"]},
        geometry=[HORIZONTAL, shapely.LineString([(1, 1), (5, 1), (5, 3), (1, 3)])],
    )
    source, destination = tmp_path / "original.gen", tmp_path / "snapped.gen"
    imod.gen.write(source, original)

    snap_gen_file_to_grid(source, destination, like=like)

    expected = snap_gen_to_grid(imod.gen.read(source), like=like).explode(ignore_index=True)
    assert_geodataframe_equal(imod.gen.read(destination), expected)
    assert imod.gen.read(destination)["id"].tolist() == ["first", "reentering", "reentering"]


def test_file_is_not_written_when_nothing_snaps_or_the_output_exists(tmp_path, gen_in, like, idfs):
    destination = tmp_path / "out.gen"
    outside = tmp_path / "outside.gen"
    outside.write_text("1\n10.0,10.0\n12.0,12.0\nEND\nEND\n", encoding="ascii")
    for extra in ({}, {"idf_top": idfs[0], "idf_bot": idfs[1]}):
        with pytest.raises(ValueError, match="No line features"):
            snap_gen_file_to_grid(outside, destination, like=like, **extra)
        assert not destination.exists()

    destination.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        snap_gen_file_to_grid(gen_in, destination, like=like)
    assert destination.read_bytes() == b"existing"
    snap_gen_file_to_grid(gen_in, destination, like=like, overwrite=True)
    assert len(imod.gen.read(destination)) == 1


def test_file_with_idfs_is_a_3d_ascii_gen_as_written_by_imod5(tmp_path, gen_in, like, idfs):
    destination = tmp_path / "out.gen"

    snap_gen_file_to_grid(gen_in, destination, like=like, idf_top=idfs[0], idf_bot=idfs[1])

    lines = destination.read_text(encoding="ascii").splitlines()
    assert lines[:7] == [
        "         1",
        "          0.000,          1.000,       1001.000",
        "          1.000,          1.000,       1001.000",
        "          1.000,          1.000,        500.000",
        "          0.000,          1.000,        500.000",
        "          0.000,          1.000,       1001.000",
        "END",
    ]
    assert lines[-1] == "END" and lines.count("END") == 5
    actual = read_ascii(destination)
    assert actual["id"].tolist() == ["1", "2", "3", "4"]
    assert actual.geometry.has_z.all()


def test_file_requires_both_idfs(tmp_path, gen_in, like, idfs):
    with pytest.raises(ValueError, match="together"):
        snap_gen_file_to_grid(gen_in, tmp_path / "out.gen", like=like, idf_top=idfs[0])
    assert not (tmp_path / "out.gen").exists()


@pytest.mark.parametrize("i3d", [False, True])
def test_cli_gensnaptogrid(tmp_path, gen_in, idfs, restore_log_handlers, i3d):

    destination = tmp_path / "out.gen"
    entries = {
        "FUNCTION": "GENSNAPTOGRID",
        "GENFILE_IN": gen_in,
        "GENFILE_OUT": destination,
        "IDFFILE": idfs[0],
    }
    if i3d:
        entries |= {"I3D": 1, "IDF_TOP": idfs[0], "IDF_BOT": idfs[1]}
    ini = tmp_path / "snap.ini"
    ini.write_text("\n".join(f"{k}={v}" for k, v in entries.items()), encoding="utf-8")

    _gensnaptogrid(ini)

    if i3d:
        assert read_ascii(destination).geometry.has_z.all()
    else:
        actual = imod.gen.read(destination)
        assert len(actual) == 1 and actual.geometry.iloc[0].length == 4.0
