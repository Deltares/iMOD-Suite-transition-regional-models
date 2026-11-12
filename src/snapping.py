"""Snap GEN line features to the edges of a structured grid.

#### How lines are snapped

The result was compared against iMOD5 5.6.1 on synthetic GEN files:

- A line selects a cell edge when it crosses the straight segment between the
  centres of the two cells that share that edge. Vertices are not moved.
- The edge of the cell that contains the last vertex of the last feature is also
  selected, even if the line stops before the centre of that cell. This is what
  iMOD5 does. It does not extend the end of any other feature, and only the
  last feature that reaches the grid counts: a last feature that is entirely
  outside the grid is ignored, but one that snaps to nothing (for example a
  line within a single cell) still prevents the extension of the one before it.
  The first vertex is not extended, so reversing a line changes the result.
- Only edges between two cells are selected: edges on the boundary of the grid
  are never snapped to. A line along the boundary, a line shorter than a cell, or
  a line outside the grid gives no output. If no line is snapped at all, no
  output file is written and the function raises an error.
- Lines are clipped to the grid. Every input feature keeps its own output line
  and attributes, also if two lines snap to the same edge.
- If a line leaves and re-enters the grid, the parts are written as separate
  features with the attributes repeated, as a GEN file has no multi-part lines.
- GEN files do not store a coordinate system, so the lines and the grid must use
  the same one.

Known difference: lines that pass exactly through cell centres (for example a
45 degree line from centre to centre) are ties. iMOD5 resolves these differently
from this implementation, and the result of this implementation can depend on
floating point rounding.

"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
import xugrid as xu
from imod.common.utilities.clip import clip_line_gdf_by_bounding_polygon
from imod.formats import gen
from imod.typing import GeoDataFrameType
from imod.typing.grid import as_ugrid_dataarray
from imod.util.spatial import spatial_reference
from scipy.spatial import cKDTree

from src.grid_helpers import create_target_grid


def _extend_line_end(line: shapely.LineString, grid) -> shapely.LineString:
    """
    Extend the last segment of a line to the centre of the cell holding its
    last vertex, as iMOD5 does: the edge of the end cell is snapped even when
    the line stops before the centre of that cell. Ends on a cell boundary or
    outside the grid are left alone.
    """
    coordinates = np.asarray(line.coords)[:, :2]
    end = coordinates[-1]
    face = grid.locate_points(end[np.newaxis])[0]
    if face < 0:
        return line
    corners = grid.node_coordinates[grid.face_node_connectivity[face]]
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    tolerance = 1e-9 * float((upper - lower).min())
    if ((end - lower) <= tolerance).any() or ((upper - end) <= tolerance).any():
        return line
    steps = np.diff(coordinates, axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    nonzero = np.flatnonzero(lengths > 0.0)
    if nonzero.size == 0:
        return line
    direction = steps[nonzero[-1]] / lengths[nonzero[-1]]
    reach = float(np.dot(grid.centroids[face] - end, direction))
    if reach <= 0.0:
        return line
    new_end = end + direction * reach * (1.0 + 1e-6)
    return shapely.LineString(np.vstack([coordinates, new_end]))


def _assert_only_line_geometries(geometry: gpd.GeoSeries) -> None:
    """Raise if any geometry is missing, empty, or not a LineString."""
    if (
        geometry.isna().any()
        or geometry.is_empty.any()
        or (geometry.geom_type != "LineString").any()
    ):
        raise ValueError("Input geometries must be non-empty LineStrings.")


def _clip_to_grid(original: GeoDataFrameType, target: xr.DataArray) -> gpd.GeoDataFrame:
    """
    Clip the lines to the extent of the grid, as 2D lines.

    Clipping explodes lines into parts. The ``position`` column holds the row of
    ``original`` each part comes from.
    """
    _, xmin, xmax, _, ymin, ymax = spatial_reference(target)
    mask = gpd.GeoDataFrame(
        geometry=[shapely.box(xmin, ymin, xmax, ymax)], crs=original.crs
    )
    lines_2d = gpd.GeoDataFrame(
        geometry=shapely.force_2d(original.geometry.to_numpy()), crs=original.crs
    )
    clipped = clip_line_gdf_by_bounding_polygon(lines_2d, mask)
    clipped["position"] = clipped.index.get_level_values(0).to_numpy()
    return clipped.reset_index(drop=True)


def _snap_vertices_to_nodes(
    lines: gpd.GeoDataFrame, grid, max_snap_distance: float
) -> gpd.GeoDataFrame:
    """
    Move vertices within ``max_snap_distance`` onto the nearest grid node and
    drop lines that collapse to a point.

    Done before snapping to edges so that collapsed lines never reach xugrid's
    edge allocator, which cannot handle zero intersected segments.
    """
    coordinates = shapely.get_coordinates(lines.geometry)
    distances, nodes = cKDTree(grid.node_coordinates).query(coordinates)
    within_distance = distances <= max_snap_distance
    coordinates[within_distance] = grid.node_coordinates[nodes[within_distance]]
    lines = lines.copy()
    lines.geometry = shapely.set_coordinates(
        lines.geometry.to_numpy().copy(), coordinates
    )
    return lines.loc[lines.geometry.length > 0.0].reset_index(drop=True)


def _extend_last_feature(
    snapped: pd.DataFrame, lines: gpd.GeoDataFrame, last_position: int, grid
) -> pd.DataFrame:
    """
    Snap the edge of the end cell of the last feature, as iMOD5 does.

    This only happens if that feature already snaps to at least one edge: a line
    within a single cell snaps to nothing, and extending it would wrongly give
    it edges. Only the extended line is snapped again.
    """
    last_rows = np.flatnonzero(lines["position"].to_numpy() == last_position)
    if last_rows.size == 0 or not np.isin(last_rows[-1], snapped["line_index"]):
        return snapped
    row = int(last_rows[-1])
    extended = gpd.GeoDataFrame(
        geometry=[_extend_line_end(lines.geometry.iloc[row], grid)], crs=lines.crs
    )
    extended_snapped = xu.create_snap_to_grid_dataframe(
        extended, grid, max_snap_distance=0.0
    )
    extended_snapped["line_index"] = row
    return pd.concat(
        [snapped[snapped["line_index"] != row], extended_snapped], ignore_index=True
    )


def _edges_to_geometries(
    snapped: pd.DataFrame, lines: gpd.GeoDataFrame, grid
) -> tuple[list[int], list[shapely.geometry.base.BaseGeometry]]:
    """
    Merge the snapped edges into one geometry per original row.

    Returns the positions of the original rows and their geometries, in input
    order. Edges hit twice by the same row are kept once.
    """
    edge_coordinates = grid.node_coordinates[grid.edge_node_connectivity]
    snapped = snapped.assign(
        position=lines["position"].to_numpy()[snapped["line_index"].to_numpy()]
    ).drop_duplicates(["position", "edge_index"])
    positions = []
    geometries = []
    for position, group in snapped.groupby("position", sort=True):
        segments = shapely.linestrings(edge_coordinates[group["edge_index"].to_numpy()])
        positions.append(position)
        geometries.append(shapely.line_merge(shapely.multilinestrings(segments)))
    return positions, geometries


def snap_gen_to_grid(
    gdf: GeoDataFrameType,
    like: xr.DataArray | None = None,
    window: tuple[float, float, float, float] | None = None,
    cellsize: float | None = None,
    max_snap_distance: float = 0.0,
) -> GeoDataFrameType:
    """
    Snap line features to the cell edges of a structured grid.

    The edge selection follows iMOD5's GENSNAPTOGRID: a line selects an edge
    when it crosses the segment joining the centres of the two cells sharing 
    that edge, and, for the last feature only, the edge of the cell holding its
    last vertex is added as well. Lines passing exactly through cell centres are 
    ties; the result can differ from iMOD5 there.

    Features are clipped to the grid extent and never merged across input rows.
    Features producing no edges are omitted. Disconnected snapped parts are
    retained as a MultiLineString in a single output row.
    Only edges between two cells are selected; exterior domain edges are omitted.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Non-empty LineString geometries and their attributes. Z values are
        ignored.
    like : xarray.DataArray, optional
        Two-dimensional grid with x and y coordinates. Data values, including
        nodata, do not affect the target grid.
    window : tuple of float, optional
        xmin, ymin, xmax, ymax. With ``like``, select cells by their centres;
        resulting cell edges can extend beyond the requested window.
    cellsize : float, optional
        Positive cell size. Required with ``window`` when ``like`` is absent.
        Cannot be combined with ``like``.
    max_snap_distance : float, default 0.0
        Maximum distance in coordinate units for moving input vertices onto
        nearby grid nodes before snapping, not a line-to-edge cutoff. iMOD5 does
        not move vertices, so the default of zero only moves vertices lying
        exactly on a node, and reproduces iMOD5. Larger values change which
        edges are selected.

    Returns
    -------
    geopandas.GeoDataFrame
        Snapped geometries with original attributes and CRS, in input order,
        with a new positional index. The input is not modified.
    """
    target = create_target_grid(like, window, cellsize)
    if not np.isfinite(max_snap_distance) or max_snap_distance < 0.0:
        raise ValueError("max_snap_distance must be finite and non-negative.")

    _assert_only_line_geometries(gdf.geometry)

    original = gdf.reset_index(drop=True)
    if original.empty:
        return original
    empty = original.iloc[:0].copy()

    grid = as_ugrid_dataarray(xr.ones_like(target, dtype=bool)).grid
    if grid.n_face == 1:
        return empty

    clipped = _clip_to_grid(original, target)
    if clipped.empty:
        return empty

    # iMOD5 only extends the end of the last feature that reaches the grid, so
    # find it before lines collapsed by vertex snapping are dropped.
    last_position = clipped["position"].max()

    lines = _snap_vertices_to_nodes(clipped, grid, max_snap_distance)
    if lines.empty:
        return empty
    snapped = xu.create_snap_to_grid_dataframe(
        lines[["geometry"]], grid, max_snap_distance=0.0
    )
    if snapped.empty:
        return empty

    snapped = _extend_last_feature(snapped, lines, last_position, grid)
    positions, geometries = _edges_to_geometries(snapped, lines, grid)

    result = original.iloc[positions].copy()
    result.geometry = geometries
    return result.reset_index(drop=True)


def snap_gen_file_to_grid(
    genfile_in: str | Path,
    genfile_out: str | Path,
    like: xr.DataArray | None = None,
    window: tuple[float, float, float, float] | None = None,
    cellsize: float | None = None,
    max_snap_distance: float = 0.0,
    overwrite: bool = False,
) -> None:
    """
    Read a GEN file, snap its 2D lines to a grid, and write a binary GEN file.

    Grid arguments have the same meaning as in :func:`snap_gen_to_grid`.
    Multipart results are written as separate LineStrings with repeated
    attributes, since GEN does not support MultiLineStrings. GEN attributes are
    stored as strings and field names are limited to eleven characters. GEN
    files do not store a CRS.

    Parameters
    ----------
    genfile_in : str or pathlib.Path
        Input ASCII or binary GEN file.
    genfile_out : str or pathlib.Path
        Output binary GEN file.
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

    Raises
    ------
    FileExistsError
        The output exists and ``overwrite`` is False.
    ValueError
        No line features produce grid edges. No output is written in this case.
    """

    destination = Path(genfile_out)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output GEN file already exists: {destination}")
    snapped = snap_gen_to_grid(
        gen.read(genfile_in),
        like=like,
        window=window,
        cellsize=cellsize,
        max_snap_distance=max_snap_distance,
    )
    if snapped.empty:
        raise ValueError("No line features could be snapped to grid edges.")
    feature_type = "feature_type" if "feature_type" in snapped.columns else None
    gen.write(destination, snapped.explode(ignore_index=True), feature_type=feature_type)
