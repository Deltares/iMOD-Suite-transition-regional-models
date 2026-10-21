# %%
"""GENPUZZLE script for connecting loose line segments in a GEN file into continuous segments."""
import argparse
from pathlib import Path

import geopandas as gpd
import imod
import numpy as np
import shapely
import xugrid as xu
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from logging_helpers import imod_logger, init_logging

DEFAULT_TOLERANCE = 0.0001


def read_gen_file(genfile_path: Path) -> gpd.GeoDataFrame:
    """
    Read GEN file and explode multipart geometries into single linestrings.
    """
    imod_logger.info(f"Reading GEN file: {genfile_path}")
    gdf = imod.formats.gen.read(genfile_path)
    return gdf.explode(index_parts=False, ignore_index=True)


def snap_loose_ends(
    lines: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Snap nearby line endpoints and return the lines with their endpoint node IDs.
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
        return lines, np.arange(2 * n_lines).reshape(2, n_lines).T
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


def _pair_junctions(node_ids: np.ndarray, endpoint_indices: np.ndarray) -> np.ndarray:
    """Pair junction endpoints in file order, leaving any odd endpoint unpaired."""
    lines_by_node = {}
    for endpoint in endpoint_indices:
        lines_by_node.setdefault(node_ids[endpoint], []).append(endpoint // 2)

    pairs = [
        pair
        for lines in lines_by_node.values()
        for pair in zip(lines[::2], lines[1::2])
    ]
    return np.array(pairs, dtype=int).reshape(-1, 2)


def _pair_lines(endpoint_nodes: np.ndarray) -> np.ndarray:
    """Pair lines at each node in file order, leaving any odd line unpaired."""
    node_ids = endpoint_nodes.ravel()
    n_endpoints = node_ids.size
    if n_endpoints == 0:
        return np.empty((0, 2), dtype=int)

    node_counts = np.bincount(node_ids)

    # Find the first and last endpoint at each node without sorting.
    endpoint_indices = np.arange(n_endpoints)
    first_endpoint = np.full(node_counts.size, n_endpoints, dtype=int)
    last_endpoint = np.full(node_counts.size, -1, dtype=int)
    np.minimum.at(first_endpoint, node_ids, endpoint_indices)
    np.maximum.at(last_endpoint, node_ids, endpoint_indices)

    # Get line index
    paired_nodes = np.flatnonzero(node_counts == 2)
    pairs = np.column_stack(
        (first_endpoint[paired_nodes], last_endpoint[paired_nodes])
    ) // 2

    # If there are junction endpoints, pair them separately.
    junction_endpoints = np.flatnonzero(node_counts[node_ids] > 2)
    if junction_endpoints.size == 0:
        return pairs

    junction_pairs = _pair_junctions(node_ids, junction_endpoints)
    return np.concatenate((pairs, junction_pairs))


def merge_lines(lines: np.ndarray, endpoint_nodes: np.ndarray) -> np.ndarray:
    """
    Merge snapped lines, pairing branches in file order at junctions.
    """
    line_pairs = _pair_lines(endpoint_nodes)
    n_lines = len(lines)

    connections = np.ones(len(line_pairs))
    first_lines = line_pairs[:, 0]
    second_lines = line_pairs[:, 1]
    line_graph = coo_matrix(
        (connections, (first_lines, second_lines)),
        shape=(n_lines, n_lines),
    )

    _, group_ids = connected_components(line_graph, directed=False)
    grouped_lines = shapely.multilinestrings(lines, indices=group_ids)
    merged_lines = shapely.line_merge(grouped_lines)

    imod_logger.info(f"Merged {n_lines} lines into {len(merged_lines)} segments")
    return merged_lines


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
    merged = merge_lines(snapped, endpoint_nodes)
    gdf_out = gpd.GeoDataFrame(geometry=merged, crs=gdf.crs)
    imod_logger.info(f"Writing GEN file: {genfile_out}")
    genfile_out.parent.mkdir(parents=True, exist_ok=True)
    imod.formats.gen.write(genfile_out, gdf_out)
    return gdf_out


def parse_args():
    """Parse command-line arguments for the GENPUZZLE script."""
    parser = argparse.ArgumentParser(
        description="Connect loose line segments in a GEN file into continuous segments"
    )
    parser.add_argument("genfile_in", type=Path, help="Path to input GEN file")
    parser.add_argument("genfile_out", type=Path, help="Path to output GEN file")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Snapping tolerance in map units (default: {DEFAULT_TOLERANCE})",
    )
    return parser.parse_args()


def main():
    """Main function for the GENPUZZLE script."""
    args = parse_args()
    init_logging()
    try:
        genpuzzle(args.genfile_in, args.genfile_out, args.tolerance)
    except Exception:
        imod_logger.exception("An error occurred during the GENPUZZLE process.")
        raise


if __name__ == "__main__":
    main()
