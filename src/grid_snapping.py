"""Snap structured-grid lines with xugrid and assemble 2D/3D wall geometry.

xugrid owns intersection and edge selection. The adapter maps its edge indices
to the structured cell indices used by the existing GEN output code. It does
not reproduce iMOD5's final-feature extension or corner/sorting quirks.

If a straight axis-aligned line exactly through cell centres selects no edges,
retry it on the nearest parallel interior grid line. Equal-distance ties go
west for vertical lines and south for horizontal lines. No boundary walls are
selected. See docs/grid_snapping/known_differences.md for measured differences.
"""

from collections.abc import Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xugrid as xu

from src.grid_helpers import GridSpec, x_nodes, y_nodes

COLUMNS = ["position", "kind", "irow", "icol"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame({name: np.array([], dtype=np.int64) for name in COLUMNS})


def _parallel_centre_line(
    coordinates: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray
) -> np.ndarray | None:
    """Project an exact straight centre line, retaining its longitudinal extent."""
    for axis, nodes in enumerate((x_edges, y_edges)):
        value = coordinates[0, axis]
        centres = (nodes[:-1] + nodes[1:]) / 2.0
        interior = np.sort(nodes[1:-1])
        if (
            interior.size
            and np.all(coordinates[:, axis] == value)
            and np.any(centres == value)
        ):
            projected = coordinates.copy()
            projected[:, axis] = interior[np.argmin(np.abs(interior - value))]
            return projected
    return None


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
    positions, coordinates, geometries = [], [], []
    for position, line in enumerate(lines):
        vertices = np.asarray(line, dtype=np.float64).reshape(-1, 2)
        if len(vertices) < 2:
            continue
        if _parallel_centre_line(vertices, x_edges, y_edges) is not None:
            # xugrid's exact-centre half-plane ties depend on traversal direction.
            vertical = np.all(vertices[:, 0] == vertices[0, 0])
            if (vertical and vertices[0, 1] > vertices[-1, 1]) or (
                not vertical and vertices[0, 0] < vertices[-1, 0]
            ):
                vertices = vertices[::-1]
        geometry = shapely.LineString(vertices)
        # Avoid xugrid's negative allocation for no intersected segments.
        if shapely.intersection(geometry, domain).length == 0.0:
            continue
        positions.append(position)
        coordinates.append(vertices)
        geometries.append(geometry)
    if not geometries:
        return _empty()

    topology = xu.Ugrid2d.from_structured_intervals1d(x_edges, y_edges)
    snapped = xu.create_snap_to_grid_dataframe(
        gpd.GeoDataFrame(geometry=geometries), topology, max_snap_distance=0.0
    )
    selected = [snapped[["line_index", "edge_index"]]]
    reported = set(snapped["line_index"])
    for index, vertices in enumerate(coordinates):
        if index in reported:
            continue
        projected = _parallel_centre_line(vertices, x_edges, y_edges)
        if projected is not None:
            retry = xu.create_snap_to_grid_dataframe(
                gpd.GeoDataFrame(geometry=[shapely.LineString(projected)]),
                topology,
                max_snap_distance=0.0,
            )
            selected.append(retry[["line_index", "edge_index"]].assign(line_index=index))
    selected_edges = pd.concat(selected, ignore_index=True).drop_duplicates()
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
