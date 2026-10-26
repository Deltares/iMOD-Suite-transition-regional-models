# %%
import argparse
import logging
import shutil
import sys
from pathlib import Path

import imod
import pandas as pd
import xarray as xr
from imod.formats.prj.prj import open_projectfile_data
from imod.logging.config import LoggerType
from imod.logging.loglevel import LogLevel
from imod.logging.pythonlogger import _formatter
from imod.mf6.ims import Solution
from imod.mf6.oc import OutputControl
from imod.mf6.simulation import Modflow6Simulation
from primod import MetaMod, MetaModDriverCoupling

from clipping import clip_wel_packages_to_domain
from settings import ConvertSettings, read_convert_settings

# Loggers in python are singletons, so this is the same instance as the one used in imod.logging.pythonlogger.PythonLogger.
imod_logger = logging.getLogger("imod")

def create_target_grid(
        window: tuple[float, float, float, float] | None, 
        cellsize: float | None
    ) -> xr.DataArray | None:
    """
    Create target grid for MODFLOW 6 simulation based on provided window and cellsize.
    """
    if window is None and cellsize is None:
        return None

    imod_logger.info(f"Creating target grid with window: {window} and cellsize: {cellsize}")
    xmin, ymin, xmax, ymax = window
    return imod.util.empty_2d(
        dx=cellsize, xmin=xmin, xmax=xmax, dy=-cellsize, ymin=ymin, ymax=ymax
    )


def convert_imod5_to_mf6_sim(
    imod5_data: dict,
    period_data: dict,
    times: list[pd.Timestamp],
    target_grid: xr.DataArray | None,
    model_name: str,
) -> Modflow6Simulation:
    """
    Convert iMOD5 data to a MODFLOW 6 simulation object.
    """
    imod_logger.info("Converting iMOD5 data to MODFLOW 6 simulation")
    mf6_sim = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        times,
        target_grid=target_grid,
        name=model_name,
    )
    # Loosen validation settings
    mf6_sim.set_validation_settings(imod.mf6.ValidationSettings(
        ignore_time=True, 
        strict_hfb_validation=False, 
        strict_well_validation=False, 
        ))
    name = f"{model_name}_model"
    # Set settings so that the simulation behaves like iMOD5
    mf6_sim[name]["oc"] = OutputControl(
        save_head="last", save_budget="last"
    )
    solution = Solution(
        modelnames=[name],
        print_option="summary",
        outer_dvclose=0.001,
        outer_maximum=150,
        inner_maximum=100,
        inner_dvclose=0.001,
        inner_rclose=100.0,
        rclose_option="strict",
        linear_acceleration="bicgstab",
        relaxation_factor=0.97,
    )
    mf6_sim["ims"] = solution

    mf6_sim[name]["npf"]["xt3d_option"] = True

    return mf6_sim


def cleanup_mf6_sim(mf6_sim: Modflow6Simulation, model_name: str) -> None:
    """
    Cleanup the MODFLOW6 simulation of erronous package data
    """
    imod_logger.info("Cleaning up MODFLOW 6 simulation")
    name = f"{model_name}_model"
    model = mf6_sim[name]
    for pkg in model.values():
        pkg.dataset.load()
    wel_keys = [key for key in model.keys() if "wel-" in key]

    # Drop wells that fall completely outside this (regional) domain's
    # extent, before mask_all_models gets a chance to error on them
    clip_wel_packages_to_domain(model, wel_keys=wel_keys)

    mask = model.domain
    mf6_sim.mask_all_models(mask, ignore_time_purge_empty=True)
    dis = model["dis"]

    topsystems_keys = [
        key
        for key in model.keys()
        if ("riv-" in key) | ("drn-" in key) | ("ghb-" in key)
    ]

    # Account for edge case where iMOD5 allocates to left-hand column of edge,
    # and iMOD Python to right-hand.
    for pkgname in wel_keys:
        model[pkgname].dataset["x"] -= 1e-10

    hfb_keys = [key for key in model.keys() if "hfb-" in key]

    for pkgname in (topsystems_keys + wel_keys + hfb_keys):
        if pkgname in model.keys():
            model[pkgname].cleanup(dis)
    # Cleanup can result in empty packages, these need to be purged.
    model.purge_empty_packages(ignore_time=True)

    for pkgname in topsystems_keys:
        model[pkgname].dataset["save_flows"] = True
    model["npf"].dataset["save_flows"] = True


def convert_imod5_to_msw_model(
    imod5_data: dict,
    mf6_sim: Modflow6Simulation,
    times: list,
    msw_dbase: Path | None,
    model_name: str,
) -> imod.msw.MetaSwapModel:
    """
    Convert iMOD5 data to a MetaSwap model.
    """
    imod_logger.info("Converting iMOD5 data to MetaSwap model")
    if "cap" not in imod5_data:
        raise ValueError("Projectfile does not contain 'cap' data, cannot create MetaSwap model.")

    name = f"{model_name}_model"
    dis_pkg = mf6_sim[name]["dis"]
    msw_model = imod.msw.MetaSwapModel.from_imod5_data(imod5_data, dis_pkg, times)
    msw_model["oc"] = imod.msw.VariableOutputControl()
    msw_model.simulation_settings["unsa_svat_path"] = msw_dbase
    msw_model.simulation_settings["vegetation_mdl"] = 1  # Simple vegetation model instead of wofost
    msw_model.simulation_settings["evapotranspiration_mdl"] = 1 # Simple evapotranspiration model
    msw_model.simulation_settings["postmsw_opt"] = 0  # Turn off PostMetaSWAP output

    return msw_model


def import_mf6_and_msw(
    imod5_data: dict,
    period_data: dict,
    is_steady_state: bool,
    msw_dbase: Path | None,
    times: list[pd.Timestamp],
    target_grid: xr.DataArray | None,
    model_name: str,
) -> tuple[Modflow6Simulation, imod.msw.MetaSwapModel | None]:
    """
    Convert iMOD5 LHM model to MODFLOW 6 using imod-python
    """
    imod_logger.info("Importing MODFLOW 6 and MetaSwap models")
    # Remove storage data for steady-state simulations, so that
    # imod.mf6.GroundwaterFlowModel.from_imod5_data adds a dummy storage package
    # again with steady-state condition on.
    if is_steady_state and ("sto" in imod5_data):
        imod5_data.pop("sto")
    # Convert to MODFLOW 6 simulation and cleanup
    mf6_simulation = convert_imod5_to_mf6_sim(
        imod5_data, period_data, times, target_grid=target_grid, model_name=model_name,
    )
    cleanup_mf6_sim(mf6_simulation, model_name)

    # Convert to MetaSwap model
    if not is_steady_state and ("cap" in imod5_data):
        msw_model = convert_imod5_to_msw_model(
            imod5_data, mf6_simulation, times, msw_dbase, model_name
        )
    else:
        msw_model = None

    return mf6_simulation, msw_model


def make_simulation(
    imod5_data: dict,
    period_data: dict,
    is_steady_state: bool,
    msw_dbase: Path | None,
    times: list[pd.Timestamp],
    target_grid: xr.DataArray | None,
    model_name: str,
) -> MetaMod | Modflow6Simulation:
    """
    Create Coupling of MODFLOW 6 and MetaSwap model
    """
    imod_logger.info("Creating simulation with MODFLOW 6 and MetaSwap coupling")
    mf6_simulation, msw_model = import_mf6_and_msw(
        imod5_data, period_data, is_steady_state, msw_dbase, times, target_grid=target_grid, model_name=model_name,
    )
    if msw_model is None:
        return mf6_simulation

    name = f"{model_name}_model"
    driver_coupling = MetaModDriverCoupling(
        mf6_model=name,
        mf6_recharge_package="msw-rch",
        mf6_wel_package="msw-sprinkling",
    )
    return MetaMod(
        msw_model,
        mf6_simulation,
        coupling_list=[driver_coupling],
    )


def init_logging() -> None:
    """Initialize logging with stream handler."""
    imod.logging.configure(
        LoggerType.PYTHON,
        log_level=LogLevel.DEBUG,
        add_default_file_handler=False,
        add_default_stream_handler=False,
    )
    # Add streamhandler
    streamhandler = logging.StreamHandler(sys.stdout)
    streamhandler.setLevel(logging.INFO)
    streamhandler.setFormatter(_formatter())
    imod_logger.addHandler(streamhandler)
    imod_logger.info("Logging initialized.")

def _add_filehandler(logfile_path: Path, log_level: int = logging.INFO):
    """Add a file handler with the specified log level to the logger."""
    filehandler = logging.FileHandler(logfile_path, mode="w")
    filehandler.setLevel(log_level)
    filehandler.setFormatter(_formatter())
    imod_logger.addHandler(filehandler)
    imod_logger.info(f"More verbose logging output will be written to {logfile_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert iMOD5 project to MODFLOW 6 and MetaSwap simulation")
    parser.add_argument("inifile", type=Path, help="Path to the ini file with settings")
    return parser.parse_args()


def main():
    args = parse_args()
    init_logging()
    try:
        # Read settings from ini file
        settings: ConvertSettings = read_convert_settings(args.inifile)
        
        # Remove existing output directory if overwrite is enabled
        if settings.overwrite and settings.out_dir.is_dir():
            shutil.rmtree(settings.out_dir, ignore_errors=True)
        settings.out_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler for logging to a file in the output directory
        logfile_path = settings.out_dir / "conversion_log.txt"
        _add_filehandler(logfile_path, settings.log_level)

        # Generate list of times for simulation
        times = settings.simulation_times()
        # Create target grid
        target_grid = create_target_grid(settings.bbox, settings.cellsize)
        # Read iMOD5 project file data
        imod5_data, period_data = open_projectfile_data(settings.prjfile_path)
        has_msw = "cap" in imod5_data
        if has_msw:
            settings.validate_msw()

        # Create coupling object
        simulation = make_simulation(
            imod5_data,
            period_data,
            settings.is_steady_state,
            settings.msw_dbase,
            times,
            target_grid=target_grid,
            model_name=settings.model_name
        )
        if isinstance(simulation, Modflow6Simulation):
            write_kwargs = {"binary": False}
            # Modify out_dir to point to a modflow6 subdirectory
            # so that it is consistent to what primod writes.
            settings.out_dir = settings.out_dir / "modflow6"
            settings.out_dir.mkdir(parents=True, exist_ok=True)
            if settings.dump_netcdf:
                netcdf_dump_dir = settings.out_dir.with_name(f"{settings.out_dir.name}_netcdf")
                imod_logger.info(f"Writing Modflow6 simulation to NetCDF format at {netcdf_dump_dir}")
                simulation.dump(netcdf_dump_dir)
        else:
            write_kwargs = {
                "modflow6_dll": settings.bin_dir/"modflow6"/"libmf6.dll",
                "metaswap_dll": settings.bin_dir/"metaswap"/"MetaSWAP.dll",
                "metaswap_dll_dependency": settings.bin_dir/"metaswap",
                "modflow6_write_kwargs": {"binary": False},
            }
        # Write coupling to disk
        simulation.write(settings.out_dir, **write_kwargs)
        # Run coupled simulation
        # simulation.run()
    except Exception:
        imod_logger.exception("An error occurred during the conversion process.") # this includes traceback in the logs
        raise

if __name__ == "__main__":
    main()
