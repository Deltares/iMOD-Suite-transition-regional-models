# %%
import argparse
from pathlib import Path

import geopandas as gpd
import imod
import numpy as np
import shapely

from logging_helpers import imod_logger, init_logging


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
    Sequentially snap each line to the endpoints of all other (already snapped)
    lines.
    """
    imod_logger.info(f"Snapping {len(lines)} lines with tolerance: {tolerance}")
    snapped = lines.copy()
    for i in range(len(snapped)):
        # Exclude own endpoints: GEOS does not snap a vertex already on a snap point.
        others = np.delete(snapped, i)
        endpoints = shapely.multipoints(
            np.concatenate(
                [shapely.get_point(others, 0), shapely.get_point(others, -1)]
            )
        )
        snapped[i] = shapely.snap(snapped[i], endpoints, tolerance)
    return snapped


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
    genfile_in: Path, genfile_out: Path, tolerance: float
) -> gpd.GeoDataFrame:
    """
    Connect loose line segments in a GEN file into continuous segments, similar
    to the GENPUZZLE batch function in iMOD5.
    """
    gdf = read_gen_file(genfile_in)
    lines = gdf.geometry.to_numpy()
    snapped = snap_loose_ends(lines, tolerance)
    merged = merge_lines(snapped)
    # Attributes cannot be mapped one-to-one after merging, only keep an id.
    gdf_out = gpd.GeoDataFrame(
        {"id": np.arange(1, len(merged) + 1)}, geometry=merged, crs=gdf.crs
    )
    imod_logger.info(f"Writing GEN file: {genfile_out}")
    genfile_out.parent.mkdir(parents=True, exist_ok=True)
    imod.formats.gen.write(genfile_out, gdf_out)
    return gdf_out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Connect loose line segments in a GEN file into continuous segments"
    )
    parser.add_argument("genfile_in", type=Path, help="Path to input GEN file")
    parser.add_argument("genfile_out", type=Path, help="Path to output GEN file")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Snapping tolerance in map units (default: 0.0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    init_logging()
    try:
        genpuzzle(args.genfile_in, args.genfile_out, args.tolerance)
    except Exception:
        imod_logger.exception("An error occurred during the GENPUZZLE process.")
        raise


if __name__ == "__main__":
    main()
