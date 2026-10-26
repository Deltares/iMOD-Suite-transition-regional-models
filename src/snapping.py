"""Snap GEN line features to the edges of a structured grid.

#### How lines are snapped

The result was compared against iMOD5 5.6.1 on synthetic GEN files:

- A line selects a cell edge when it crosses the straight segment between the
  centres of the two cells that share that edge. Vertices are not moved.
- The edge of the cell that contains the last vertex of a line is also selected,
  even if the line stops before the centre of that cell. This is what iMOD5 does.
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
import shapely
import xarray as xr
import xugrid as xu
from imod.common.utilities.clip import clip_line_gdf_by_bounding_polygon
from imod.formats.gen import read, write
from imod.typing import GeoDataFrameType
from imod.typing.grid import as_ugrid_dataarray
from imod.util.spatial import empty_2d, spatial_reference
from scipy.spatial import cKDTree


def _target_grid(
    like: xr.DataArray | None,
    window: tuple[float, float, float, float] | None,
    cellsize: float | None,
) -> xr.DataArray:
    if window is not None:
        xmin, ymin, xmax, ymax = window
        if not np.isfinite(window).all() or xmin >= xmax or ymin >= ymax:
            raise ValueError(
                "window must contain finite xmin, ymin, xmax, ymax with "
                "xmin < xmax and ymin < ymax."
            )

    if like is not None:
        if cellsize is not None:
            raise ValueError("cellsize cannot be combined with like.")
        if like.ndim != 2 or set(like.dims) != {"x", "y"}:
            raise ValueError(
                "like must be a two-dimensional grid with x and y dimensions."
            )
        if window is not None:
            x = like["x"]
            y = like["y"]
            like = like.sel(
                x=x[(x >= xmin) & (x <= xmax)],
                y=y[(y >= ymin) & (y <= ymax)],
            )
        if like.sizes["x"] == 0 or like.sizes["y"] == 0:
            raise ValueError(
                "window must select at least one grid cell in each dimension."
            )
        return like

    if window is None or cellsize is None:
        raise ValueError("Provide like, or both window and cellsize.")
    if not np.isfinite(cellsize) or cellsize <= 0.0:
        raise ValueError("cellsize must be finite and greater than zero.")
    return empty_2d(
        dx=cellsize,
        xmin=xmin,
        xmax=xmax,
        dy=-cellsize,
        ymin=ymin,
        ymax=ymax,
    )


def _extend_line_ends(lines: np.ndarray, grid) -> np.ndarray:
    """
    Extend the last segment of each line to the centre of the cell holding its
    last vertex, as iMOD5 does: the edge of the end cell is snapped even when
    the line stops before the centre of that cell. Starts are not extended.
    Ends on a cell boundary or outside the grid are left alone.
    """
    ends = np.array([line.coords[-1][:2] for line in lines])
    faces = grid.locate_points(ends)
    extended = lines.copy()
    for i, (line, end, face) in enumerate(zip(lines, ends, faces)):
        if face < 0:
            continue
        corners = grid.node_coordinates[grid.face_node_connectivity[face]]
        lower, upper = corners.min(axis=0), corners.max(axis=0)
        tolerance = 1e-9 * float((upper - lower).min())
        if ((end - lower) <= tolerance).any() or ((upper - end) <= tolerance).any():
            continue
        coordinates = np.asarray(line.coords)[:, :2]
        steps = np.diff(coordinates, axis=0)
        lengths = np.hypot(steps[:, 0], steps[:, 1])
        nonzero = np.flatnonzero(lengths > 0.0)
        if nonzero.size == 0:
            continue
        direction = steps[nonzero[-1]] / lengths[nonzero[-1]]
        reach = float(np.dot(grid.centroids[face] - end, direction))
        if reach <= 0.0:
            continue
        new_end = end + direction * reach * (1.0 + 1e-6)
        extended[i] = shapely.LineString(np.vstack([coordinates, new_end]))
    return extended


def snap_gen_to_grid(
    gdf: GeoDataFrameType,
    like: xr.DataArray | None = None,
    window: tuple[float, float, float, float] | None = None,
    cellsize: float | None = None,
    max_snap_distance: float = 0.0,
) -> GeoDataFrameType:
    """
    Snap 2D line features to the cell edges of a structured grid.

    The edge selection follows iMOD5's GENSNAPTOGRID: a line selects an edge
    when it crosses the segment joining the centres of the two cells sharing 
    that edge, and the edge of the cell holding the last vertex of a line is 
    added as well. Lines passing exactly through cell centres are ties; the 
    result can differ from iMOD5 there.

    Features are clipped to the grid extent and never merged across input rows.
    Features producing no edges are omitted. Disconnected snapped parts are
    retained as a MultiLineString in a single output row.
    Only edges between two cells are selected; exterior domain edges are omitted.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Non-empty 2D LineString geometries and their attributes.
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
    target = _target_grid(like, window, cellsize)
    if not np.isfinite(max_snap_distance) or max_snap_distance < 0.0:
        raise ValueError("max_snap_distance must be finite and non-negative.")

    geometry = gdf.geometry
    if (
        geometry.isna().any()
        or geometry.is_empty.any()
        or (geometry.geom_type != "LineString").any()
        or shapely.has_z(geometry).any()
    ):
        raise ValueError("Input geometries must be non-empty 2D LineStrings.")

    original = gdf.reset_index(drop=True)
    if original.empty:
        return original

    active = xr.ones_like(target, dtype=bool)
    grid = as_ugrid_dataarray(active).grid
    if grid.n_face == 1:
        return original.iloc[:0].copy()
    edge_coordinates = grid.node_coordinates[grid.edge_node_connectivity]

    _, xmin, xmax, _, ymin, ymax = spatial_reference(target)
    mask = gpd.GeoDataFrame(
        geometry=[shapely.box(xmin, ymin, xmax, ymax)], crs=original.crs
    )
    clipped = clip_line_gdf_by_bounding_polygon(
        original.geometry.to_frame(name="geometry"), mask
    )
    if clipped.empty:
        return original.iloc[:0].copy()

    # Snap vertices first so lines collapsed to a node never reach xugrid's
    # edge allocator, which cannot handle zero intersected segments.
    coordinates = shapely.get_coordinates(clipped.geometry)
    distances, nodes = cKDTree(grid.node_coordinates).query(coordinates)
    within_distance = distances <= max_snap_distance
    coordinates[within_distance] = grid.node_coordinates[nodes[within_distance]]
    clipped.geometry = shapely.set_coordinates(
        clipped.geometry.to_numpy().copy(), coordinates
    )
    clipped = clipped.loc[clipped.geometry.length > 0.0]
    if clipped.empty:
        return original.iloc[:0].copy()

    # Clipping explodes parts; xugrid indices refer to these rows, not originals.
    original_positions = clipped.index.get_level_values(0).to_numpy()
    lines = clipped.reset_index(drop=True)
    snapped = xu.create_snap_to_grid_dataframe(lines, grid, max_snap_distance=0.0)
    if snapped.empty:
        return original.iloc[:0].copy()

    # iMOD5 also snaps the edge of the end cell, but only for lines that snap
    # to at least one edge: a line within a single cell snaps to nothing.
    with_edges = np.unique(snapped["line_index"].to_numpy())
    geometries_array = lines.geometry.to_numpy().copy()
    geometries_array[with_edges] = _extend_line_ends(
        geometries_array[with_edges], grid
    )
    lines.geometry = geometries_array
    snapped = xu.create_snap_to_grid_dataframe(lines, grid, max_snap_distance=0.0)

    snapped["line_index"] = original_positions[snapped["line_index"].to_numpy()]
    snapped = snapped.drop_duplicates(["line_index", "edge_index"])

    positions = []
    geometries = []
    for position, group in snapped.groupby("line_index", sort=True):
        segments = shapely.linestrings(edge_coordinates[group["edge_index"].to_numpy()])
        positions.append(position)
        geometries.append(shapely.line_merge(shapely.multilinestrings(segments)))

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
        read(genfile_in),
        like=like,
        window=window,
        cellsize=cellsize,
        max_snap_distance=max_snap_distance,
    )
    if snapped.empty:
        raise ValueError("No line features could be snapped to grid edges.")
    feature_type = "feature_type" if "feature_type" in snapped.columns else None
    write(destination, snapped.explode(ignore_index=True), feature_type=feature_type)
