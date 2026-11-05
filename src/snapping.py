"""Snap GEN line features to the edges of a structured grid.

#### How lines are snapped

Edge selection uses the public xugrid snapping API, adapted in
``src/grid_snapping.py``. Differences from iMOD5 5.6.1 are accepted and
illustrated in ``docs/grid_snapping/known_differences.md``:

- xugrid selects edges through intersected line segments and neighbouring cell
  centres. Vertices are not moved by default, and features are independent of
  file order. The legacy final-feature extension is not reproduced.
- Outside segments do not receive the legacy corner-pairing treatment.
- Otherwise-empty straight axis-aligned lines exactly through cell centres are
  retried on the nearest parallel interior grid line, choosing west/south on
  ties. Short lines can still produce no edges.
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
closed polygon with Z values. See :func:`src.grid_snapping.vertical_walls` for the
rules, which were established by comparing with iMOD5 5.6.1. iMOD5 averages TOP
and BOT grids with smaller cells onto the target grid. That is not supported
here: TOP and BOT must be on the same grid as the target grid, or on a larger
grid with the same cells.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from imod.formats import gen
from imod.typing import GeoDataFrameType

from src.grid_helpers import (
    GridSpec,
    cell_values,
    create_target_grid,
    grid_spec,
    open_idf_grid,
    sample_on_grid,
    x_nodes,
    y_nodes,
)
from src.grid_snapping import (
    snap_lines_to_edges,
    unique_walls,
    vertical_walls,
    wall_end_points,
)


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

    Edge selection uses xugrid, with a parallel-grid-line retry for otherwise
    empty straight cell-centre lines; see the module docstring.

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
        neither must be given. They must contain every cell of the target grid,
        with the same cell size; nodata is NaN. They may cover a larger area.

    Returns
    -------
    geopandas.GeoDataFrame
        Snapped geometries with original attributes and CRS, in input order
        (raster order for walls), with a new positional index. The input is not
        modified. Empty if nothing snaps.

    Raises
    ------
    ValueError
        Invalid grid arguments, geometries or ``max_snap_distance``; only one of
        ``top`` and ``bottom`` is given; or ``top`` or ``bottom`` is not on the
        grid of the target.
    """
    if (top is None) != (bottom is None):
        raise ValueError("top and bottom must be given together.")
    target = create_target_grid(like, window, cellsize)
    if not np.isfinite(max_snap_distance) or max_snap_distance < 0.0:
        raise ValueError("max_snap_distance must be finite and non-negative.")
    _assert_only_line_geometries(gdf.geometry)
    if top is not None:
        top_values = cell_values(sample_on_grid(top, target, "top"))
        bottom_values = cell_values(sample_on_grid(bottom, target, "bottom"))

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
        IDFs with the top and the bottom of the walls: a single layer, on the
        grid that is snapped to (see :func:`snap_gen_to_grid`). Either both or
        neither must be given.

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
