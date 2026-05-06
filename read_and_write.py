# %%
import configparser
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import xarray as xr
import imod
from imod.formats.prj.prj import open_projectfile_data
from imod.logging.config import LoggerType
from imod.logging.loglevel import LogLevel
from imod.mf6.ims import Solution
from imod.mf6.oc import OutputControl
from imod.mf6.simulation import Modflow6Simulation
import pandas as pd
from primod import MetaMod, MetaModDriverCoupling

@dataclass
class Settings:
    prjfile_path: Path
    msw_dbase: Path
    out_dir: Path
    bin_dir: Path
    start_date: str
    end_date: str
    interval: str
    cellsize: Optional[float]
    bbox: Optional[str]
    model_name: str


def read_settings(inifile: Path) -> Settings:
    config = configparser.ConfigParser(allow_unnamed_section=True)
    config.read(inifile)

    section = configparser.UNNAMED_SECTION
    return Settings(
        prjfile_path=Path(config.get(section, "PRJFILE_IN")),
        msw_dbase=Path(config.get(section, "MSW_DBASE")),
        out_dir=Path(config.get(section, "OUTPUT_FOLDER")),
        bin_dir=Path(config.get(section, "COUPLER_DIR")),
        start_date=config.get(section, "SDATE"),
        end_date=config.get(section, "EDATE"),
        interval=config.get(section, "INTERVAL", fallback="D"),
        cellsize=config.getfloat(section, "CELLSIZE", fallback=None),
        bbox=config.get(section, "WINDOW", fallback=None),
        model_name=config.get(section, "MODELNAME", fallback="imported"),
    )


def create_target_grid(
        window: Optional[str], 
        cellsize: Optional[float]
    ) -> Optional[xr.DataArray]:
    """
    Create target grid for MODFLOW 6 simulation based on provided window and cellsize.
    """

    if window is None and cellsize is None:
        return None
    elif window is None or cellsize is None:
        raise ValueError("Both window and cellsize must be provided to create target grid.")

    window_tuple = tuple(map(float, window.split(",")))
    xmin, ymin, xmax, ymax = window_tuple

    return imod.util.empty_2d(
        dx=cellsize, xmin=xmin, xmax=xmax, dy=-cellsize, ymin=ymin, ymax=ymax
    )

def convert_imod5_to_mf6_sim(
    imod5_data: dict, period_data: dict, times: list[pd.Timestamp], target_grid: Optional[xr.DataArray], model_name: str
) -> Modflow6Simulation:
    """
    Convert iMOD5 data to a MODFLOW 6 simulation object.
    """
    simulation = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        times,
        target_grid=target_grid,
        name=model_name,
    )
    # Loosen validation settings
    simulation.set_validation_settings(imod.mf6.ValidationSettings(
        ignore_time=True, 
        strict_hfb_validation=False, 
        strict_well_validation=False, 
        ))
    name = f"{model_name}_model"
    # Set settings so that the simulation behaves like iMOD5
    simulation[name]["oc"] = OutputControl(
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
    simulation["ims"] = solution

    simulation[name]["npf"]["xt3d_option"] = True

    return simulation


def cleanup_mf6_sim(simulation: Modflow6Simulation, model_name: str) -> None:
    """
    Cleanup the MODFLOW6 simulation of erronous package data
    """
    name = f"{model_name}_model"
    model = simulation[name]
    for pkg in model.values():
        pkg.dataset.load()

    mask = model.domain
    simulation.mask_all_models(mask, ignore_time_purge_empty=True)
    dis = model["dis"]

    topsystems_keys = [
        key
        for key in model.keys()
        if ("riv-" in key) | ("drn-" in key) | ("ghb-" in key)
    ]

    wel_keys = [key for key in model.keys() if "wel-" in key]
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
    msw_dbase: Path | str,
    model_name: str,
) -> imod.msw.MetaSwapModel:
    """
    Convert iMOD5 data to a MetaSwap model.
    """
    name = f"{model_name}_model"
    dis_pkg = mf6_sim[name]["dis"]
    msw_model = imod.msw.MetaSwapModel.from_imod5_data(imod5_data, dis_pkg, times)
    msw_model["oc"] = imod.msw.VariableOutputControl()
    msw_model.simulation_settings["unsa_svat_path"] = msw_dbase
    msw_model.simulation_settings["vegetation_mdl"] = (
        1  # Simple vegetation model instead of wofost
    )
    msw_model.simulation_settings["evapotranspiration_mdl"] = (
        1  # Simple evapotranspiration model
    )
    msw_model.simulation_settings["postmsw_opt"] = 0  # Turn off PostMetaSWAP output

    return msw_model


def import_mf6_and_msw(
    prjfile_path: Path, msw_dbase: Path, times: list[pd.Timestamp], target_grid: xr.DataArray, model_name: str
) -> tuple[Modflow6Simulation, imod.msw.MetaSwapModel]:
    """
    Convert iMOD5 LHM model to MODFLOW 6 using imod-python
    """

    # Read iMOD5 project file data
    imod5_data, period_data = open_projectfile_data(prjfile_path)

    # Convert to MODFLOW 6 simulation and cleanup
    mf6_simulation = convert_imod5_to_mf6_sim(
        imod5_data, period_data, times, target_grid=target_grid, model_name=model_name,
    )
    cleanup_mf6_sim(mf6_simulation, model_name)

    # Convert to MetaSwap model
    msw_model = convert_imod5_to_msw_model(
        imod5_data, mf6_simulation, times, msw_dbase, model_name
    )

    return mf6_simulation, msw_model


def make_metamod_coupling(
    prjfile_path: Path, msw_dbase: Path, times: list[pd.Timestamp], target_grid: xr.DataArray, model_name: str
) -> MetaMod:
    """
    Create Coupling of MODFLOW 6 and MetaSwap model
    """
    mf6_simulation, msw_model = import_mf6_and_msw(
        prjfile_path, msw_dbase, times, target_grid=target_grid, model_name=model_name,
    )
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


# %%
if __name__ == "__main__":
    # Read settings from ini file
    inifile = sys.argv[1]

    settings = read_settings(inifile)
    out_dir = settings.out_dir
    bin_dir = settings.bin_dir

    # Configure logging to file in output directory
    logfile_path = out_dir / "conversion_log.txt"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )

        # Generate list of times for simulation
        times = pd.date_range(start=settings.start_date, end=settings.end_date, freq=settings.interval).tolist()
        # Create target grid
        target_grid = create_target_grid(settings.bbox, settings.cellsize)
        # Create coupling object
        coupling = make_metamod_coupling(settings.prjfile_path, settings.msw_dbase, times, target_grid=target_grid, model_name=settings.model_name)
        # Write coupling to disk
        coupling.write(
            out_dir,
            modflow6_dll=bin_dir/"msw.dll",
            metaswap_dll=bin_dir,
            metaswap_dll_dependency=bin_dir/"mf6.dll",
        )
        # Run coupled simulation
        # coupling.run()
