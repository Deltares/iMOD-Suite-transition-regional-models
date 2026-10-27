# %%
"""GENPUZZLE script for connecting loose line segments in a GEN file into continuous segments."""
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import imod
import numpy as np
import shapely
import xugrid as xu

from src.log import imod_logger

DEFAULT_TOLERANCE = 0.001


def read_gen_file(genfile_path: Path) -> gpd.GeoDataFrame:
    """
    Read GEN file and explode multipart geometries into single linestrings.
    """
    imod_logger.info(f"Reading GEN file: {genfile_path}")
    gdf = imod.formats.gen.read(genfile_path)
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    # Snapping and merging work in 2D, so drop any Z values.
    geometries = gdf.geometry.force_2d().to_numpy()
    # imod reads closed features as polygons. Treat them as chains, as iMOD5 does,
    # so their endpoints take part in snapping and joining.
    is_polygon = np.array([g.geom_type == "Polygon" for g in geometries], dtype=bool)
    geometries[is_polygon] = [
        shapely.LineString(shapely.get_coordinates(g.exterior))
        for g in geometries[is_polygon]
    ]
    return gdf.set_geometry(geometries)


def snap_loose_ends(
    lines: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Return snapped lines and endpoint node IDs, or None if no snapping occurred.
    """
    imod_logger.info(f"Snapping {len(lines)} lines with tolerance: {tolerance}")
    n_lines = len(lines)

    # Extract endpoints of all lines for snapping
    endpoints_xy = np.concatenate(
        [
            shapely.get_coordinates(shapely.get_point(lines, 0)),
            shapely.get_coordinates(shapely.get_point(lines, -1)),
        ]
    )
    # Snap endpoints to a common node
    endpoint_to_node, x_snapped, y_snapped = xu.snap_nodes(
        endpoints_xy[:, 0], endpoints_xy[:, 1], tolerance
    )
    if endpoint_to_node is None:
        return lines, None

    snapped_xy = np.column_stack([x_snapped, y_snapped])[endpoint_to_node]

    # Update the coordinates of the line endpoints with the snapped positions
    coords = shapely.get_coordinates(lines)
    n_coords = shapely.get_num_coordinates(lines)
    last = np.cumsum(n_coords) - 1
    first = last - n_coords + 1
    coords[first] = snapped_xy[:n_lines]
    coords[last] = snapped_xy[n_lines:]
    return (
        shapely.set_coordinates(lines.copy(), coords),
        endpoint_to_node.reshape(2, n_lines).T,
    )


def _ending_at(
    coords: np.ndarray, end_nodes: tuple[int, int], node: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return the chain reversed if needed so that it ends at `node`."""
    if end_nodes.index(node) == 0:
        return coords[::-1], end_nodes[::-1]
    return coords, end_nodes


def _starting_at(
    coords: np.ndarray, end_nodes: tuple[int, int], node: int
) -> tuple[np.ndarray, tuple[int, int]]:
    """Return the chain reversed if needed so that it starts at `node`."""
    if end_nodes.index(node) == 1:
        return coords[::-1], end_nodes[::-1]
    return coords, end_nodes


def _join_chains(
    chain_coords: np.ndarray,
    chain_end_nodes: tuple[int, int],
    connected_coords: np.ndarray,
    connected_end_nodes: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Join two chains that share an endpoint node into one chain.

    For example, if the chain runs A to B and the connected line runs C to A, the
    shared node A is the start of the chain, so the connected line goes first:
    C to A to B. If the connected line runs A to C instead, it is reversed first,
    giving the same result.
    """
    if chain_end_nodes[0] in connected_end_nodes:
        # The shared node is the chain's start, so the connected line goes first.
        front_coords, front_end_nodes = _ending_at(
            connected_coords, connected_end_nodes, chain_end_nodes[0]
        )
        back_coords, back_end_nodes = chain_coords, chain_end_nodes
    else:
        # The shared node is the chain's end, so the connected line goes last.
        front_coords, front_end_nodes = chain_coords, chain_end_nodes
        back_coords, back_end_nodes = _starting_at(
            connected_coords, connected_end_nodes, chain_end_nodes[1]
        )
    return np.vstack([front_coords, back_coords[1:]]), (
        front_end_nodes[0],
        back_end_nodes[1],
    )


def _find_next_connected_line(
    line_idx: int,
    chains: dict[int, tuple[np.ndarray, tuple[int, int]]],
    lines_at_node: dict[int, set[int]],
) -> int | None:
    """
    Return the position of the first line after `line_idx` in the list passed to
    `_merge_connected_lines` that touches it at an endpoint.

    Lines that have been absorbed are no longer in `chains`, so they are skipped.
    Returns None if there is none.
    """
    _, end_nodes = chains[line_idx]
    connected = {
        connected_idx
        for node in end_nodes
        for connected_idx in lines_at_node[node]
        if connected_idx > line_idx and connected_idx in chains
    }
    return min(connected, default=None)


def _merge_connected_lines(lines: np.ndarray, endpoint_nodes: np.ndarray) -> np.ndarray:
    """
    Join lines that touch at an endpoint, following the GENPUZZLE join in iMOD5.

    `chains` holds the lines not yet absorbed, keyed by position. Lines are handled
    in order. Each line keeps joining the first connected line after it until none
    are left, and each joined line is removed from `chains`.

    This is being done instead of using shapely.merge_linestrings to mimic the 
    behavior of how GENPUZZLE function in iMOD5 handles junctions closely as possible.
    """
    chains = {
        line_idx: (shapely.get_coordinates(line), (int(start), int(end)))
        for line_idx, (line, (start, end)) in enumerate(zip(lines, endpoint_nodes))
    }

    lines_at_node: dict[int, set[int]] = defaultdict(set)
    for line_idx, (_, end_nodes) in chains.items():
        for node in end_nodes:
            lines_at_node[node].add(line_idx)

    for line_idx in range(len(lines)):
        if line_idx not in chains:
            continue
        connected_idx = _find_next_connected_line(line_idx, chains, lines_at_node)
        while connected_idx is not None:
            chain_coords, chain_end_nodes = chains[line_idx]
            connected_coords, connected_end_nodes = chains.pop(connected_idx)
            chains[line_idx] = _join_chains(
                chain_coords, chain_end_nodes, connected_coords, connected_end_nodes
            )
            connected_idx = _find_next_connected_line(line_idx, chains, lines_at_node)

    return np.array(
        [shapely.LineString(coords) for coords, _ in chains.values()],
        dtype=object,
    )


def _rings_to_polygons(geometries: np.ndarray) -> np.ndarray:
    """Convert closed lines to polygons, starting at the top-left vertex."""
    result = []
    for geometry in geometries:
        if shapely.is_closed(geometry):
            ring = shapely.get_coordinates(geometry)[:-1]
            start = np.lexsort((-ring[:, 1], ring[:, 0]))[0]
            ring = np.roll(ring, -start, axis=0)
            geometry = shapely.Polygon(np.vstack([ring, ring[:1]]))
        result.append(geometry)
    return np.array(result, dtype=object)


def genpuzzle(
    genfile_in: Path, genfile_out: Path, tolerance: float = DEFAULT_TOLERANCE
) -> gpd.GeoDataFrame:
    """
    Connect loose line segments in a GEN file into continuous segments, following 
    the GENPUZZLE batch function in iMOD5.
    """
    gdf = read_gen_file(genfile_in)
    lines = gdf.geometry.to_numpy()
    snapped, endpoint_nodes = snap_loose_ends(lines, tolerance)
    if endpoint_nodes is None:
        merged = snapped
    else:
        merged = _merge_connected_lines(snapped, endpoint_nodes)
    imod_logger.info(f"Merged {len(lines)} lines into {len(merged)} segments")
    gdf_out = gpd.GeoDataFrame(geometry=_rings_to_polygons(merged))
    imod_logger.info(f"Writing GEN file: {genfile_out}")
    genfile_out.parent.mkdir(parents=True, exist_ok=True)
    imod.formats.gen.write(genfile_out, gdf_out)
    return gdf_out

