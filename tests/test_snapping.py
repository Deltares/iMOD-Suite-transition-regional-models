import geopandas as gpd
import imod
import numpy as np
import pytest
import shapely
import xarray as xr
from geopandas.testing import assert_geodataframe_equal
from imod.typing.grid import as_ugrid_dataarray
from imod.util import empty_2d

from src.snapping import _target_grid, snap_gen_file_to_grid, snap_gen_to_grid


@pytest.fixture
def like():
    return empty_2d(dx=1.0, xmin=0.0, xmax=4.0, dy=-1.0, ymin=0.0, ymax=4.0)


def test_target_grid_like(like):
    xr.testing.assert_identical(_target_grid(like, None, None), like)


def test_target_grid_window_and_cellsize(like):
    actual = _target_grid(None, (0.0, 0.0, 4.0, 4.0), 1.0)
    xr.testing.assert_identical(actual, like)


def test_target_grid_like_and_window(like):
    actual = _target_grid(like, (1.0, 1.0, 3.0, 3.0), None)
    xr.testing.assert_identical(actual, like.sel(x=slice(1.0, 3.0), y=slice(3.0, 1.0)))


@pytest.mark.parametrize(
    "window, cellsize",
    [(None, None), (None, 1.0), ((0.0, 0.0, 4.0, 4.0), None)],
)
def test_target_grid_missing_inputs(window, cellsize):
    with pytest.raises(ValueError, match="window and cellsize"):
        _target_grid(None, window, cellsize)


@pytest.mark.parametrize("cellsize", [0.0, -1.0, np.nan, np.inf])
def test_target_grid_invalid_cellsize(cellsize):
    with pytest.raises(ValueError, match="cellsize"):
        _target_grid(None, (0.0, 0.0, 4.0, 4.0), cellsize)


@pytest.mark.parametrize(
    "window",
    [
        (4.0, 0.0, 0.0, 4.0),
        (0.0, 4.0, 4.0, 0.0),
        (0.0, 0.0, 0.0, 4.0),
        (0.0, 0.0, np.nan, 4.0),
        (0.0, 0.0, 4.0, np.inf),
    ],
)
def test_target_grid_invalid_window(window):
    with pytest.raises(ValueError, match="window"):
        _target_grid(None, window, 1.0)


def test_target_grid_like_and_cellsize(like):
    with pytest.raises(ValueError, match="cellsize"):
        _target_grid(like, None, 2.0)


def test_target_grid_window_outside_like(like):
    with pytest.raises(ValueError, match="window"):
        _target_grid(like, (10.0, 10.0, 12.0, 12.0), None)


def test_target_grid_requires_two_dimensions(like):
    with pytest.raises(ValueError, match="two-dimensional"):
        _target_grid(like.expand_dims(layer=[1]), None, None)


@pytest.mark.parametrize(
    "coordinates",
    [
        [(0.1, 1.1), (3.9, 1.1)],
        [(1.1, 0.1), (1.1, 3.9)],
        [(1.1, 1.1), (2.9, 2.9)],
        [(-1.0, 1.1), (5.0, 1.1)],
        [(1.0, 1.0), (3.0, 3.0)],
    ],
    ids=["horizontal", "vertical", "diagonal", "crossing", "through-nodes"],
)
def test_snap_gen_to_grid_edges(like, coordinates):
    original = gpd.GeoDataFrame(
        {"id": ["line"], "value": [3.0]},
        geometry=[shapely.LineString(coordinates)],
        crs="EPSG:28992",
    )
    actual = snap_gen_to_grid(original, like=like)

    assert len(actual) == 1
    assert actual.crs == original.crs
    assert actual["id"].tolist() == ["line"]
    assert actual["value"].tolist() == [3.0]
    line = actual.geometry.iloc[0]
    assert line.geom_type == "LineString"
    xy = np.asarray(line.coords)
    np.testing.assert_allclose(xy, np.round(xy))
    delta = np.abs(np.diff(xy, axis=0))
    np.testing.assert_allclose(delta.sum(axis=1), 1.0)
    assert ((xy >= 0.0) & (xy <= 4.0)).all()
    if coordinates == [(1.1, 1.1), (2.9, 2.9)]:
        assert line.length == 4.0
        assert set(map(tuple, xy[[0, -1]])) == {(1.0, 1.0), (3.0, 3.0)}
        assert (delta[1:] != delta[:-1]).all()


def test_snap_gen_to_grid_preserves_separate_features(like):
    line = shapely.LineString([(0.1, 1.1), (3.9, 1.1)])
    original = gpd.GeoDataFrame(
        {"id": ["outside", "first", "second"], "value": [0, 1, 2]},
        index=[7, 7, 12],
        geometry=[shapely.LineString([(10, 10), (12, 12)]), line, line],
        crs="EPSG:28992",
    )
    unchanged = original.copy(deep=True)
    actual = snap_gen_to_grid(original, like=like)

    assert actual["id"].tolist() == ["first", "second"]
    assert actual["value"].tolist() == [1, 2]
    assert actual.geometry.iloc[0].equals(actual.geometry.iloc[1])
    assert_geodataframe_equal(original, unchanged)


def test_snap_gen_to_grid_reentering_line(like):
    original = gpd.GeoDataFrame(
        {"id": ["reentering"]},
        geometry=[shapely.LineString([(1, 1), (5, 1), (5, 3), (1, 3)])],
    )
    actual = snap_gen_to_grid(original, like=like)
    assert actual["id"].tolist() == ["reentering"]
    assert actual.geometry.iloc[0].geom_type == "MultiLineString"
    assert len(actual.geometry.iloc[0].geoms) == 2


@pytest.mark.parametrize("outside", [False, True])
def test_snap_gen_to_grid_empty(like, outside):
    original = gpd.GeoDataFrame(
        {"id": ["outside"] if outside else []},
        geometry=[shapely.LineString([(10, 10), (12, 12)])] if outside else [],
        crs="EPSG:28992",
    )
    actual = snap_gen_to_grid(original, like=like)
    assert actual.empty
    assert actual.columns.tolist() == original.columns.tolist()
    assert actual.crs == original.crs


@pytest.mark.parametrize("distance", [-1.0, np.nan, np.inf])
def test_snap_gen_to_grid_invalid_distance(like, distance):
    original = gpd.GeoDataFrame(geometry=[shapely.LineString([(1, 1), (3, 3)])])
    with pytest.raises(ValueError, match="max_snap_distance"):
        snap_gen_to_grid(original, like=like, max_snap_distance=distance)


@pytest.mark.parametrize(
    "geometry",
    [
        shapely.Point(1, 1),
        shapely.LineString([(1, 1, 0), (3, 3, 0)]),
        shapely.LineString(),
        None,
    ],
)
def test_snap_gen_to_grid_rejects_invalid_geometry(like, geometry):
    original = gpd.GeoDataFrame(geometry=[geometry])
    with pytest.raises(ValueError, match="non-empty 2D LineString"):
        snap_gen_to_grid(original, like=like)


def test_snap_gen_to_grid_public_api(like):
    original = gpd.GeoDataFrame(geometry=[shapely.LineString([(1.0, 1.0), (3.0, 3.0)])])
    assert not snap_gen_to_grid(original, like=like).empty


def test_snap_gen_to_grid_short_line(like):
    original = gpd.GeoDataFrame(
        {"id": ["short"]},
        geometry=[shapely.LineString([(1.01, 1.01), (1.02, 1.02)])],
    )
    actual = snap_gen_to_grid(original, like=like)
    assert actual.empty
    assert actual.columns.tolist() == original.columns.tolist()


@pytest.mark.parametrize("scale", [1.0, 100.0])
def test_snap_gen_to_grid_distance_units(scale):
    original = gpd.GeoDataFrame(
        geometry=[shapely.LineString(np.array([(1.1, 1.1), (2.9, 2.9)]) * scale)]
    )
    kwargs = {"window": (0.0, 0.0, 4.0 * scale, 4.0 * scale), "cellsize": scale}
    default = snap_gen_to_grid(original, **kwargs)
    explicit = snap_gen_to_grid(original, max_snap_distance=0.0, **kwargs)
    assert_geodataframe_equal(default, explicit)
    assert default.geometry.iloc[0].length == 4.0 * scale


def _unit_edges(gdf):
    edges = set()
    for part in shapely.get_parts(gdf.geometry.to_numpy()):
        xy = np.round(np.asarray(part.coords)).astype(int)
        edges |= {frozenset(map(tuple, pair)) for pair in zip(xy[:-1], xy[1:])}
    return edges


def test_snap_gen_to_grid_does_not_move_vertices_by_default(like):
    # Moving the vertices to the nodes (0, 0) and (4, 1) would give a different
    # set of edges. Expected edges as written by iMOD5 5.6.1.
    original = gpd.GeoDataFrame(
        geometry=[shapely.LineString([(0.2, 0.3), (3.8, 1.4)])]
    )
    actual = snap_gen_to_grid(original, like=like)
    expected = {
        frozenset({(1, 0), (1, 1)}),
        frozenset({(1, 1), (2, 1)}),
        frozenset({(2, 1), (3, 1)}),
        frozenset({(3, 1), (4, 1)}),
    }
    assert _unit_edges(actual) == expected


@pytest.mark.parametrize(
    "coordinates, n_edges",
    [
        # End 0.2 before the cell centre: iMOD5 still snaps the end cell's edge.
        ([(1.1, 0.1), (1.1, 2.3)], 3),
        ([(1.1, 0.1), (1.1, 2.45)], 3),
        # End after the cell centre: edge already crossed.
        ([(1.1, 0.1), (1.1, 2.7)], 3),
        # Only the end is extended: the start 0.2 before the centre is a start.
        ([(1.1, 2.3), (1.1, 0.1)], 2),
        # Reversed line, end at 1.7 in a cell with centre 1.5: extended.
        ([(1.1, 3.9), (1.1, 1.7)], 3),
        ([(0.1, 1.1), (2.3, 1.1)], 3),
        ([(2.3, 1.1), (0.1, 1.1)], 2),
    ],
)
def test_snap_gen_to_grid_extends_line_end_to_cell_centre(like, coordinates, n_edges):
    original = gpd.GeoDataFrame(geometry=[shapely.LineString(coordinates)])
    actual = snap_gen_to_grid(original, like=like)
    assert len(_unit_edges(actual)) == n_edges


def test_snap_gen_to_grid_does_not_extend_line_inside_single_cell(like):
    original = gpd.GeoDataFrame(
        geometry=[shapely.LineString([(1.1, 1.1), (1.2, 1.1)])]
    )
    assert snap_gen_to_grid(original, like=like).empty


def test_snap_gen_to_grid_zero_distance(like):
    original = gpd.GeoDataFrame(geometry=[shapely.LineString([(1.0, 1.0), (3.0, 3.0)])])
    assert not snap_gen_to_grid(original, like=like, max_snap_distance=0.0).empty


def test_snap_gen_file_to_grid_roundtrip(tmp_path, like):
    original = gpd.GeoDataFrame(
        {"id": ["first", "second"], "name": ["a", "b"]},
        geometry=[
            shapely.LineString([(0.1, 1.1), (3.9, 1.1)]),
            shapely.LineString([(1.1, 0.1), (1.1, 3.9)]),
        ],
    )
    source = tmp_path / "original.gen"
    destination = tmp_path / "snapped.gen"
    imod.gen.write(source, original)

    snap_gen_file_to_grid(source, destination, like=like)

    expected = snap_gen_to_grid(imod.gen.read(source), like=like)
    assert_geodataframe_equal(imod.gen.read(destination), expected)


def test_snap_gen_file_to_grid_multipart(tmp_path, like):
    source = tmp_path / "original.gen"
    destination = tmp_path / "snapped.gen"
    original = gpd.GeoDataFrame(
        {"id": ["reentering"]},
        geometry=[shapely.LineString([(1, 1), (5, 1), (5, 3), (1, 3)])],
    )
    imod.gen.write(source, original)

    snap_gen_file_to_grid(source, destination, like=like)

    actual = imod.gen.read(destination)
    assert actual["id"].tolist() == ["reentering", "reentering"]
    assert (actual.geometry.geom_type == "LineString").all()


def test_snap_gen_file_to_grid_empty(tmp_path, like):
    source = tmp_path / "original.gen"
    destination = tmp_path / "snapped.gen"
    original = gpd.GeoDataFrame(
        {"id": ["outside"]},
        geometry=[shapely.LineString([(10, 10), (12, 12)])],
    )
    imod.gen.write(source, original)

    with pytest.raises(ValueError, match="No line features"):
        snap_gen_file_to_grid(source, destination, like=like)
    assert not destination.exists()


def test_snap_gen_file_to_grid_overwrite(tmp_path, like):
    source = tmp_path / "original.gen"
    destination = tmp_path / "snapped.gen"
    original = gpd.GeoDataFrame(
        {"id": ["line"]},
        geometry=[shapely.LineString([(0.1, 1.1), (3.9, 1.1)])],
    )
    imod.gen.write(source, original)
    destination.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        snap_gen_file_to_grid(source, destination, like=like)
    assert destination.read_bytes() == b"existing"

    snap_gen_file_to_grid(source, destination, like=like, overwrite=True)
    assert len(imod.gen.read(destination)) == 1


def test_snap_gen_to_grid_custom_geometry_name(like):
    original = gpd.GeoDataFrame(
        {"id": ["line"]},
        geometry=[shapely.LineString([(1.1, 1.1), (2.9, 2.9)])],
    ).rename_geometry("line_geometry")
    actual = snap_gen_to_grid(original, like=like)
    assert actual.geometry.name == "line_geometry"
    assert actual["id"].tolist() == ["line"]


def test_snap_gen_to_grid_single_cell():
    original = gpd.GeoDataFrame(geometry=[shapely.LineString([(0.1, 0.1), (0.9, 0.9)])])
    actual = snap_gen_to_grid(
        original, window=(0.0, 0.0, 1.0, 1.0), cellsize=1.0
    )
    # xugrid only selects edges separating two cell centres.
    assert actual.empty


@pytest.mark.parametrize(
    "coordinates",
    [
        [(0.0, 0.0), (4.0, 0.0)],
        [(4.0, 0.0), (4.0, 4.0)],
        [(4.0, 4.0), (0.0, 4.0)],
        [(0.0, 4.0), (0.0, 0.0)],
    ],
)
def test_snap_gen_to_grid_domain_boundary(like, coordinates):
    original = gpd.GeoDataFrame(geometry=[shapely.LineString(coordinates)])
    actual = snap_gen_to_grid(original, like=like)
    assert actual.empty


def test_target_grid_offcentre_window(like):
    actual = _target_grid(like, (0.6, 0.6, 3.4, 3.4), None)
    np.testing.assert_array_equal(actual.x, [1.5, 2.5])
    np.testing.assert_array_equal(actual.y, [2.5, 1.5])


def test_snap_gen_file_to_grid_ascii(tmp_path, like):
    source = tmp_path / "original.gen"
    source.write_text("1\n0.1,1.1\n3.9,1.1\nEND\nEND\n", encoding="ascii")
    destination = tmp_path / "snapped.gen"

    snap_gen_file_to_grid(source, destination, like=like)

    actual = imod.gen.read(destination)
    assert len(actual) == 1
    assert actual.geometry.iloc[0].geom_type == "LineString"
    assert "feature_typ" not in actual.columns


@pytest.mark.parametrize(
    "dimension, coordinates",
    [
        ("x", [(0.1, 1.1), (3.9, 1.1)]),
        ("y", [(1.1, 0.1), (1.1, 3.9)]),
    ],
)
def test_snap_gen_to_grid_singleton_dimension(like, dimension, coordinates):
    original = gpd.GeoDataFrame(geometry=[shapely.LineString(coordinates)])
    actual = snap_gen_to_grid(original, like=like.isel({dimension: [1]}))
    assert not actual.empty
    assert actual.geometry.length.sum() == 1.0


def test_snap_gen_to_grid_nonuniform_grid():
    like = xr.DataArray(
        np.ones((3, 3)),
        dims=("y", "x"),
        coords={
            "x": [0.5, 2.0, 3.5],
            "y": [3.5, 2.0, 0.5],
            "dx": ("x", [1.0, 2.0, 1.0]),
            "dy": ("y", [-1.0, -2.0, -1.0]),
        },
    )
    original = gpd.GeoDataFrame(geometry=[shapely.LineString([(0.1, 1.1), (3.9, 1.1)])])
    actual = snap_gen_to_grid(original, like=like)
    grid = as_ugrid_dataarray(like).grid
    edges = shapely.multilinestrings(grid.node_coordinates[grid.edge_node_connectivity])

    assert not actual.empty
    assert edges.covers(actual.geometry.iloc[0])
