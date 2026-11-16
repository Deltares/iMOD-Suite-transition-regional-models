"""Snap GEN line features to the edges of a structured grid.

#### How lines are snapped

Edge selection uses the public xugrid snapping API. Differences from iMOD5
5.6.1 are accepted and illustrated in ``docs/grid_snapping/known_differences.md``:

- xugrid selects edges through intersected line segments and neighbouring cell
  centres. Vertices are not moved by default, and features are independent of
  file order. The legacy final-feature extension is not reproduced.
- Outside segments do not receive the legacy corner-pairing treatment.
- A straight line that exactly follows cell centres can select no edges; this
  hardly occurs outside synthetic examples.
- Only edges between two cells are selected: edges on the boundary of the grid
  are never snapped to. A line along the boundary, a line shorter than a cell, or
  a line outside the grid gives no output. If no line is snapped at all, no
  output file is written and the function raises an error.
- Polygons are snapped as their exterior ring, as iMOD5 treats rings as lines.
- GEN files do not store a coordinate system, so the lines and the grid must use
  the same one.
- Equidistant and nonuniform structured grids use the same xugrid algorithm.

#### 3D walls (``I3D=1``)

With ``top`` and ``bottom`` grids every snapped edge becomes a vertical wall: a
closed polygon with Z values. See :func:`vertical_walls` for the rules, which
were established by comparing with iMOD5 5.6.1. TOP and BOT are regridded to
the target grid with the area-weighted mean, as iMOD5 does for smaller cells.
"""

from collections.abc import Sequence
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xugrid as xu
from imod.formats import gen
from imod.typing import GeoDataFrameType

from src.grid_helpers import (
    GridSpec,
    cell_values,
    create_target_grid,
    grid_spec,
    open_idf_grid,
    regrid_to_target,
    x_nodes,
    y_nodes,
)

COLUMNS = ["position", "kind", "irow", "icol"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({name: np.array([], dtype=np.int64) for name in COLUMNS})


def snap_lines_to_edges(lines: Sequence[np.ndarray], grid: GridSpec) -> pd.DataFrame:
    """Select interior walls with xugrid, once per input feature and edge.

    Columns are ``position`` (input feature index), ``kind`` (0 for the east
    wall, 1 for the south wall), and zero-based ``irow`` and ``icol``. Row zero
    is northernmost. Features are independent of file order; within each
    feature walls are returned in raster order.
    """
    if grid.nrow == 1 and grid.ncol == 1:
        return _empty()
    x_edges, y_edges = x_nodes(grid), y_nodes(grid)
    domain = shapely.box(x_edges[0], y_edges[-1], x_edges[-1], y_edges[0])
    positions, geometries = [], []
    for position, line in enumerate(lines):
        vertices = np.asarray(line, dtype=np.float64).reshape(-1, 2)
        if len(vertices) < 2:
            continue
        geometry = shapely.LineString(vertices)
        # Avoid xugrid's negative allocation for no intersected segments.
        if shapely.intersection(geometry, domain).length == 0.0:
            continue
        positions.append(position)
        geometries.append(geometry)
    if not geometries:
        return _empty()

    topology = xu.Ugrid2d.from_structured_intervals1d(x_edges, y_edges)
    snapped = xu.create_snap_to_grid_dataframe(
        gpd.GeoDataFrame(geometry=geometries), topology, max_snap_distance=0.0
    )
    selected_edges = snapped[["line_index", "edge_index"]].drop_duplicates()
    if selected_edges.empty:
        return _empty()

    edge_index = selected_edges["edge_index"].to_numpy()
    neighbours = topology.edge_face_connectivity[edge_index]
    interior = (neighbours >= 0).all(axis=1)
    selected_edges = selected_edges[interior]
    neighbours = neighbours[interior]
    endpoints = topology.node_coordinates[
        topology.edge_node_connectivity[edge_index[interior]]
    ]
    # Structured faces are row-major; the first neighbour is north or west.
    cell = neighbours.min(axis=1)
    vertical = endpoints[:, 0, 0] == endpoints[:, 1, 0]
    result = pd.DataFrame({
        "position": np.asarray(positions)[selected_edges["line_index"].to_numpy()],
        "kind": (~vertical).astype(np.int64),
        "irow": cell // grid.ncol,
        "icol": cell % grid.ncol,
    }, columns=COLUMNS).astype(np.int64)
    return result.sort_values(COLUMNS, kind="stable").reset_index(drop=True)


def wall_end_points(grid: GridSpec, edges: pd.DataFrame) -> np.ndarray:
    """Wall endpoints, north-to-south or west-to-east, shape ``(n, 2, 2)``."""
    x_edges = x_nodes(grid)
    y_edges = y_nodes(grid)
    kind = edges["kind"].to_numpy()
    irow = edges["irow"].to_numpy()
    icol = edges["icol"].to_numpy()
    vertical = kind == 0
    result = np.empty((len(edges), 2, 2))
    result[:, :, 0] = np.where(
        vertical[:, np.newaxis],
        x_edges[icol + 1][:, np.newaxis],
        np.column_stack([x_edges[icol], x_edges[icol + 1]]),
    )
    result[:, :, 1] = np.where(
        vertical[:, np.newaxis],
        np.column_stack([y_edges[irow], y_edges[irow + 1]]),
        y_edges[irow + 1][:, np.newaxis],
    )
    return result


def raster_order(edges: pd.DataFrame) -> np.ndarray:
    """Sort by row, column, then vertical before horizontal wall, as iMOD5."""
    return np.lexsort(
        (edges["kind"].to_numpy(), edges["icol"].to_numpy(), edges["irow"].to_numpy())
    )


def unique_walls(edges: pd.DataFrame) -> pd.DataFrame:
    """Shared walls keep the first feature's attributes and use raster order."""
    unique = edges.drop_duplicates(["kind", "irow", "icol"], keep="first")
    return unique.iloc[raster_order(unique)].reset_index(drop=True)


def vertical_walls(
    grid: GridSpec, edges: pd.DataFrame, top: np.ndarray, bottom: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """Construct closed vertical polygons using the existing iMOD5 output rules.

    ``edges`` contains unique walls in raster order. TOP is the maximum of the
    two adjacent cells; BOT is their minimum. Drop walls with nodata in either
    cell. Zero and negative heights remain valid output. Rings are A-top,
    B-top, B-bottom, A-bottom, A-top.
    """
    irow = edges["irow"].to_numpy()
    icol = edges["icol"].to_numpy()
    kind = edges["kind"].to_numpy()
    other_row = irow + kind
    other_col = icol + 1 - kind
    ztop = np.maximum(top[irow, icol], top[other_row, other_col])
    zbottom = np.minimum(bottom[irow, icol], bottom[other_row, other_col])

    keep = np.isfinite(ztop) & np.isfinite(zbottom)
    kept = edges[keep].reset_index(drop=True)
    if kept.empty:
        return kept, np.empty(0, dtype=object)
    first, second = wall_end_points(grid, kept).transpose(1, 0, 2)
    ztop, zbottom = ztop[keep], zbottom[keep]
    dx = second[:, 0] - first[:, 0]
    dy = second[:, 1] - first[:, 1]
    swap = np.where(np.abs(dx) > np.abs(dy), dx < 0.0, dy > 0.0)[:, np.newaxis]
    a = np.where(swap, second, first)
    b = np.where(swap, first, second)
    rings = np.stack(
        [
            np.column_stack([a, ztop]),
            np.column_stack([b, ztop]),
            np.column_stack([b, zbottom]),
            np.column_stack([a, zbottom]),
            np.column_stack([a, ztop]),
        ],
        axis=1,
    )
    return kept, shapely.polygons(shapely.linearrings(rings))


def _assert_only_line_geometries(geometry: gpd.GeoSeries) -> None:
    """Raise if any geometry is missing, empty, or neither a LineString nor a Polygon."""
    if (
        geometry.isna().any()
        or geometry.is_empty.any()
        or (~geometry.geom_type.isin(["LineString", "Polygon"])).any()
    ):
        raise ValueError("Input geometries must be non-empty LineStrings or Polygons.")


def _vertices(geometry: gpd.GeoSeries) -> list[np.ndarray]:
    """The 2D vertices of each line, or of the exterior ring of each polygon."""
    return [
        shapely.get_coordinates(
            geom.exterior if geom.geom_type == "Polygon" else geom
        )
        for geom in geometry
    ]


def _nearest_node(nodes: np.ndarray, values: np.ndarray) -> np.ndarray:
    """The value of ``nodes`` (sorted ascending) closest to each of ``values``."""
    upper = np.clip(np.searchsorted(nodes, values), 1, len(nodes) - 1)
    lower = upper - 1
    use_lower = np.abs(values - nodes[lower]) <= np.abs(nodes[upper] - values)
    return np.where(use_lower, nodes[lower], nodes[upper])


def _snap_vertices_to_nodes(
    lines: list[np.ndarray], grid: GridSpec, max_snap_distance: float
) -> list[np.ndarray]:
    """
    Move vertices within ``max_snap_distance`` onto the nearest grid node, and
    empty the lines that collapse to a point.
    """
    counts = [len(line) for line in lines]
    coordinates = np.concatenate(lines) if lines else np.empty((0, 2))
    x_edges = x_nodes(grid)
    y_edges = y_nodes(grid)
    nearest = np.column_stack(
        [
            _nearest_node(x_edges, coordinates[:, 0]),
            _nearest_node(y_edges[::-1], coordinates[:, 1]),
        ]
    )
    within_distance = np.hypot(*(nearest - coordinates).T) <= max_snap_distance
    coordinates = np.where(within_distance[:, np.newaxis], nearest, coordinates)
    moved = np.split(coordinates, np.cumsum(counts)[:-1])
    empty = np.empty((0, 2))
    return [line if (line != line[0]).any() else empty for line in moved]


def _merged_lines(
    grid: GridSpec, edges: pd.DataFrame
) -> tuple[list[int], list[shapely.geometry.base.BaseGeometry]]:
    """
    Merge the snapped edges into one geometry per original row.

    Returns the positions of the original rows and their geometries, in input
    order.
    """
    end_points = wall_end_points(grid, edges)
    positions = []
    geometries = []
    for position, group in edges.groupby("position", sort=True):
        segments = shapely.linestrings(end_points[group.index.to_numpy()])
        positions.append(position)
        geometries.append(shapely.line_merge(shapely.multilinestrings(segments)))
    return positions, geometries


def snap_gen_to_grid(
    gdf: GeoDataFrameType,
    like: xr.DataArray | None = None,
    window: tuple[float, float, float, float] | None = None,
    cellsize: float | None = None,
    max_snap_distance: float = 0.0,
    top: xr.DataArray | None = None,
    bottom: xr.DataArray | None = None,
) -> GeoDataFrameType:
    """
    Snap line features to the cell edges of a structured grid.

    Edge selection uses xugrid; see the module docstring.

    Without ``top`` and ``bottom`` features are never merged across input rows.
    Features producing no edges are omitted. Disconnected snapped parts are
    retained as a MultiLineString in a single output row. Only edges between two
    cells are selected; exterior domain edges are omitted.

    With ``top`` and ``bottom`` (``I3D=1`` of iMOD5) each snapped edge becomes a
    closed vertical polygon with Z values instead, written once however many
    features snap to it, in iMOD5's raster order, with the attributes of the
    first feature that snapped to it.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Non-empty LineString or Polygon geometries and their attributes. Z
        values are ignored and polygons are snapped as their exterior ring.
    like : xarray.DataArray, optional
        Two-dimensional grid with x and y coordinates, equidistant or not. Data
        values, including nodata, do not affect the target grid.
    window : tuple of float, optional
        xmin, ymin, xmax, ymax. Requires ``cellsize``: the grid starts at the
        lower-left corner and drops the part of the window that does not fill a
        cell, as iMOD5 does.
    cellsize : float, optional
        Positive cell size. Required with ``window`` when ``like`` is absent.
        Cannot be combined with ``like``.
    max_snap_distance : float, default 0.0
        Maximum distance in coordinate units for moving input vertices onto
        nearby grid nodes before snapping, not a line-to-edge cutoff. iMOD5 does
        not move vertices, so the default of zero does not move any vertex and
        avoids vertex pre-snapping. Larger values change which edges are selected.
    top, bottom : xarray.DataArray, optional
        ``IDF_TOP`` and ``IDF_BOT``: two-dimensional grids with ``x`` and ``y``
        coordinates that give the top and bottom of the walls. Either both or
        neither must be given. They are regridded to the target grid with the
        area-weighted mean, so they may have other cell sizes and extent; cells
        without a valid value are nodata (NaN) and give no wall.

    Returns
    -------
    geopandas.GeoDataFrame
        Snapped geometries with original attributes and CRS, in input order
        (raster order for walls), with a new positional index. The input is not
        modified. Empty if nothing snaps.

    Raises
    ------
    ValueError
        Invalid grid arguments, geometries or ``max_snap_distance``, or only one
        of ``top`` and ``bottom`` is given.
    """
    if (top is None) != (bottom is None):
        raise ValueError("top and bottom must be given together.")
    target = create_target_grid(like, window, cellsize)
    if not np.isfinite(max_snap_distance) or max_snap_distance < 0.0:
        raise ValueError("max_snap_distance must be finite and non-negative.")
    _assert_only_line_geometries(gdf.geometry)
    if top is not None:
        top_values = cell_values(regrid_to_target(top, target, "top"))
        bottom_values = cell_values(regrid_to_target(bottom, target, "bottom"))

    original = gdf.reset_index(drop=True)
    empty = original.iloc[:0].copy()
    if original.empty:
        return empty

    grid = grid_spec(target)
    lines = _vertices(original.geometry)
    if max_snap_distance > 0.0:
        lines = _snap_vertices_to_nodes(lines, grid, max_snap_distance)
    edges = snap_lines_to_edges(lines, grid)
    if edges.empty:
        return empty

    if top is None:
        positions, geometries = _merged_lines(grid, edges)
    else:
        edges, geometries = vertical_walls(
            grid, unique_walls(edges), top_values, bottom_values
        )
        if edges.empty:
            return empty
        positions = edges["position"].to_numpy()
    result = original.iloc[positions].copy()
    result.geometry = geometries
    return result.reset_index(drop=True)


def _write_gen_ascii_3d(path: Path, walls: gpd.GeoDataFrame) -> None:
    """
    Write the vertical walls of ``snap_gen_to_grid`` as a 3D ASCII GEN file: ids
    1 to N, then x, y, z for each of the five vertices, as iMOD5 lays it out.

    Binary GEN files cannot hold Z values.

    TODO: move to iMOD Python, which should read and write 3D GEN files.
    """
    rings = shapely.get_coordinates(walls.geometry.exterior, include_z=True)
    lines = []
    for number, ring in enumerate(rings.reshape(-1, 5, 3), start=1):
        lines.append(f"{number:10d}")
        lines.extend(",".join(f"{value:15.3f}" for value in vertex) for vertex in ring)
        lines.append("END")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def snap_gen_file_to_grid(
    genfile_in: str | Path,
    genfile_out: str | Path,
    like: xr.DataArray | None = None,
    window: tuple[float, float, float, float] | None = None,
    cellsize: float | None = None,
    max_snap_distance: float = 0.0,
    overwrite: bool = False,
    idf_top: str | Path | None = None,
    idf_bot: str | Path | None = None,
) -> None:
    """
    Read a GEN file, snap its lines to a grid, and write a GEN file.

    Grid arguments have the same meaning as in :func:`snap_gen_to_grid`.

    Without ``idf_top`` and ``idf_bot`` the output is a binary GEN file with the
    merged lines of :func:`snap_gen_to_grid` and the attributes of the input.
    Multipart results are written as separate LineStrings with repeated
    attributes, since GEN does not support MultiLineStrings. GEN attributes are
    stored as strings and field names are limited to eleven characters.

    With ``idf_top`` and ``idf_bot`` (``I3D=1``) the output is a 3D ASCII GEN
    file, as binary GEN files cannot hold Z values. It has one vertical polygon
    per snapped edge, with the ids 1 to N as iMOD5 writes them: the ids and
    attributes of the input are not written.

    GEN files do not store a CRS.

    Parameters
    ----------
    genfile_in : str or pathlib.Path
        Input ASCII or binary GEN file.
    genfile_out : str or pathlib.Path
        GEN file to write.
    like : xarray.DataArray, optional
        Target grid, for example a two-dimensional IDF read with ``imod.idf.open``.
    window : tuple of float, optional
        xmin, ymin, xmax, ymax.
    cellsize : float, optional
        Cell size when no target grid is supplied.
    max_snap_distance : float, default 0.0
        Maximum distance for moving vertices onto nearby grid nodes before
        snapping. See :func:`snap_gen_to_grid`.
    overwrite : bool, default False
        Allow replacing an existing output file.
    idf_top, idf_bot : str or pathlib.Path, optional
        IDFs with the top and the bottom of the walls: a single layer, regridded
        to the grid that is snapped to (see :func:`snap_gen_to_grid`). Either both
        or neither must be given.

    Raises
    ------
    FileExistsError
        The output exists and ``overwrite`` is False.
    ValueError
        No line features produce grid edges, or only one of ``idf_top`` and
        ``idf_bot`` is given. No output is written in this case.
    """
    destination = Path(genfile_out)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output GEN file already exists: {destination}")
    top = None if idf_top is None else open_idf_grid(idf_top)
    bottom = None if idf_bot is None else open_idf_grid(idf_bot)
    snapped = snap_gen_to_grid(
        gen.read(genfile_in),
        like=like,
        window=window,
        cellsize=cellsize,
        max_snap_distance=max_snap_distance,
        top=top,
        bottom=bottom,
    )
    if snapped.empty:
        raise ValueError("No line features could be snapped to grid edges.")
    if top is not None:
        _write_gen_ascii_3d(destination, snapped)
        return
    feature_type = "feature_type" if "feature_type" in snapped.columns else None
    gen.write(destination, snapped.explode(ignore_index=True), feature_type=feature_type)
