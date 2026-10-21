# %%
"""GENPUZZLE script for connecting loose line segments in a GEN file into continuous segments."""
import argparse
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import imod
import numpy as np
import shapely
import xugrid as xu
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from logging_helpers import imod_logger, init_logging

DEFAULT_TOLERANCE = 0.001


def read_gen_file(genfile_path: Path) -> gpd.GeoDataFrame:
    """
    Read GEN file and explode multipart geometries into single linestrings.
    """
    imod_logger.info(f"Reading GEN file: {genfile_path}")
    gdf = imod.formats.gen.read(genfile_path)
    return gdf.explode(index_parts=False, ignore_index=True)


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


def _pair_lines(endpoint_nodes: np.ndarray) -> np.ndarray:
    """Pair lines at each node in file order, leaving any odd line unpaired."""
    lines_by_node = defaultdict(list)
    for line_index, nodes in enumerate(endpoint_nodes):
        for node in nodes:
            lines_by_node[node].append(line_index)

    pairs = []
    for line_indices in lines_by_node.values():
        pairs.extend(zip(line_indices[::2], line_indices[1::2]))

    return np.array(pairs, dtype=int).reshape(-1, 2)


def _merge_with_junctions(lines: np.ndarray, endpoint_nodes: np.ndarray) -> np.ndarray:
    """Merge a network containing junctions, pairing branches in file order."""
    # Select which lines connect at each endpoint node.
    line_pairs = _pair_lines(endpoint_nodes)
    n_lines = len(lines)

    # Build a graph with one vertex per line and one edge per selected pair.
    first_lines, second_lines = line_pairs.T
    line_graph = coo_matrix(
        (np.ones(len(line_pairs)), (first_lines, second_lines)),
        shape=(n_lines, n_lines),
    )

    # Identify groups connected through one or more pairs.
    _, group_ids = connected_components(line_graph, directed=False)
    # Collect each group's geometries and merge their connected segments.
    grouped_lines = shapely.multilinestrings(lines, indices=group_ids)
    return shapely.line_merge(grouped_lines)


def merge_lines(lines: np.ndarray, endpoint_nodes: np.ndarray) -> np.ndarray:
    """Merge chains directly, using file-order pairing when junctions exist."""
    node_counts = np.bincount(endpoint_nodes.ravel())
    if np.any(node_counts > 2):
        # Note: For some reason, all the nodes need to be passed otherwise 
        # we lose the xugrid ids of what is connected
        merged_lines = _merge_with_junctions(lines, endpoint_nodes)
    else:
        merged = shapely.line_merge(shapely.multilinestrings(lines))
        merged_lines = shapely.get_parts(merged)

    imod_logger.info(f"Merged {len(lines)} lines into {len(merged_lines)} segments")
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
    if endpoint_nodes is None:
        merged = snapped
    else:
        merged = merge_lines(snapped, endpoint_nodes)
    gdf_out = gpd.GeoDataFrame(geometry=merged)
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
