# %%
import argparse
import shutil
from pathlib import Path

from src.convert import convert_model
from src.genpuzzle import genpuzzle
from src.log import add_filehandler, imod_logger, init_logging
from src.settings import (
    ConvertSettings,
    GenSnapToGridSettings,
    read_convert_settings,
    read_gen_snap_to_grid_settings,
    read_genpuzzle_settings,
)
from src.snapping import snap_gen_file_to_grid

_CHOICES = ["convert", "gensnaptogrid", "genpuzzle"]
def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI for executing iMOD-Suite transition regional models functions"
    )
    parser.add_argument(
        "function", 
        type=str, 
        choices=_CHOICES,
        help="Function to execute",
    )
    parser.add_argument("inifile", type=Path, help="Path to the ini file with settings")
    return parser.parse_args()

def _convert(ini_file: Path):
    settings: ConvertSettings = read_convert_settings(ini_file)
    if settings.overwrite and settings.out_dir.is_dir():
        shutil.rmtree(settings.out_dir, ignore_errors=True)
    settings.out_dir.mkdir(parents=True, exist_ok=True)

    logfile_path = settings.out_dir / "conversion_log.txt"
    add_filehandler(logfile_path, settings.log_level)

    convert_model(settings)

def _gensnaptogrid(ini_file: Path):
    settings: GenSnapToGridSettings = read_gen_snap_to_grid_settings(ini_file)
    add_filehandler(
        settings.genfile_in.with_name("gensnaptogrid.log"), settings.log_level
    )
    snap_gen_file_to_grid(
        genfile_in=settings.genfile_in,
        genfile_out=settings.genfile_out,
        like=settings.like,
        window=settings.window,
        cellsize=settings.cellsize,
        max_snap_distance=settings.max_snap_distance,
        overwrite=settings.overwrite,
    )

def _genpuzzle(ini_file: Path):
    settings = read_genpuzzle_settings(ini_file)
    add_filehandler(
        settings.genfile_in.with_name("genpuzzle.log"), settings.log_level
    )
    genpuzzle(settings.genfile_in, settings.genfile_out, settings.tolerance)


def main():
    args = parse_args()
    init_logging()

    try:
        match args.function:
            case "convert":
                _convert(args.inifile)
            case "gensnaptogrid":
                _gensnaptogrid(args.inifile)
            case "genpuzzle":
                _genpuzzle(args.inifile)
            case _:
                raise ValueError(f"Unknown function specified, must be one of {_CHOICES}, got {args.function}.")

    except Exception:
        # this includes traceback in the logs
        imod_logger.exception("An unexpected error occurred.")
        raise


if __name__ == "__main__":
    main()
