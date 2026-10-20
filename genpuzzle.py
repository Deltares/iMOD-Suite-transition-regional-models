# %%
"""GENPUZZLE script for connecting loose line segments in a GEN file into continuous segments."""
import argparse
from pathlib import Path

import geopandas as gpd
import imod
import numpy as np
import shapely
import xugrid as xu

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
) -> np.ndarray:
    """
    Snap line endpoints within tolerance of each other onto a single node.
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
        return lines
    snapped_xy = np.column_stack([x_snapped, y_snapped])[endpoint_to_node]

    # Update the coordinates of the line endpoints with the snapped positions
    coords = shapely.get_coordinates(lines)
    n_coords = shapely.get_num_coordinates(lines)
    last = np.cumsum(n_coords) - 1
    first = last - n_coords + 1
    coords[first] = snapped_xy[:n_lines]
    coords[last] = snapped_xy[n_lines:]
    return shapely.set_coordinates(lines.copy(), coords)


def merge_lines(lines: np.ndarray) -> np.ndarray:
    """
    Merge snapped lines that share endpoints into continuous linestrings.
    """
    imod_logger.info("Merging snapped lines into continuous segments")
    merged = shapely.line_merge(shapely.multilinestrings(lines))
    merged_parts = shapely.get_parts(merged)
    imod_logger.info(
        f"Merged {len(lines)} lines into {len(merged_parts)} segments"
    )
    return merged_parts


def genpuzzle(
    genfile_in: Path, genfile_out: Path, tolerance: float = DEFAULT_TOLERANCE
) -> gpd.GeoDataFrame:
    """
    Connect loose line segments in a GEN file into continuous segments, following 
    the GENPUZZLE batch function in iMOD5.
    """
    gdf = read_gen_file(genfile_in)
    lines = gdf.geometry.to_numpy()
    snapped = snap_loose_ends(lines, tolerance)
    merged = merge_lines(snapped)
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
