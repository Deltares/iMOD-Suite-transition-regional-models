# %%
import configparser
import sys
from pathlib import Path
from typing import Optional

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


def create_target_grid(
        window: Optional[str], 
        cellsize: Optional[float]
    ) -> Optional[xr.DataArray]:

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
    imod5_data: dict, period_data: dict, times: list[pd.Timestamp], target_grid: Optional[xr.DataArray]
) -> Modflow6Simulation:
    simulation = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        times,
        target_grid=target_grid,
    )
    # Loosen validation settings
    simulation.set_validation_settings(imod.mf6.ValidationSettings(
        ignore_time=True, 
        strict_hfb_validation=False, 
        strict_well_validation=False, 
        ))
    # Set settings so that the simulation behaves like iMOD5
    simulation["imported_model"]["oc"] = OutputControl(
        save_head="last", save_budget="last"
    )
    solution = Solution(
        modelnames=["imported_model"],
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

    simulation["imported_model"]["npf"]["xt3d_option"] = True

    return simulation


def cleanup_mf6_sim(simulation: Modflow6Simulation) -> None:
    """
    Cleanup the simulation of erronous package data
    """
    model = simulation["imported_model"]
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
) -> imod.msw.MetaSwapModel:
    dis_pkg = mf6_sim["imported_model"]["dis"]
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


def import_lhm_mf6_and_msw(
    prjfile_path: Path, msw_dbase: Path, times: list[pd.Timestamp], target_grid: xr.DataArray
) -> tuple[Modflow6Simulation, imod.msw.MetaSwapModel]:
    """
    Convert iMOD5 LHM model to MODFLOW 6 using imod-python
    """

    # Read iMOD5 project file data
    imod5_data, period_data = open_projectfile_data(prjfile_path)

    # Convert to MODFLOW 6 simulation and cleanup
    mf6_simulation = convert_imod5_to_mf6_sim(
        imod5_data, period_data, times, target_grid=target_grid
    )
    cleanup_mf6_sim(mf6_simulation)

    # Convert to MetaSwap model
    msw_model = convert_imod5_to_msw_model(
        imod5_data, mf6_simulation, times, msw_dbase
    )

    return mf6_simulation, msw_model


def make_lhm_coupling(
    prjfile_path: Path, msw_dbase: Path, times: list[pd.Timestamp], target_grid: xr.DataArray
) -> MetaMod:
    """
    Test coupling of LHM MODFLOW 6 and MetaSwap models
    """
    mf6_simulation, msw_model = import_lhm_mf6_and_msw(
        prjfile_path, msw_dbase, times, target_grid=target_grid
    )

    driver_coupling = MetaModDriverCoupling(
        mf6_model="imported_model",
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

    config = configparser.ConfigParser(allow_unnamed_section=True)
    config.read(inifile)

    section = configparser.UNNAMED_SECTION
    prj_path = Path(config.get(section, "PRJFILE_IN"))
    msw_dbase = Path(config.get(section, "MSW_DBASE"))
    out_dir = Path(config.get(section, "OUTPUT_FOLDER"))
    bin_dir = Path(config.get(section, "COUPLER_DIR"))
    start_date = config.get(section, "SDATE")
    end_date = config.get(section, "EDATE")
    interval = config.get(section, "INTERVAL", fallback="D")
    cellsize = config.getfloat(section, "CELLSIZE", fallback=None)
    bbox = config.get(section, "WINDOW", fallback=None)

    # Generate list of times for simulation
    times = pd.date_range(start=start_date, end=end_date, freq=interval).tolist()

    #Path management
    logfile_path = out_dir / "conversion_log.txt"

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )
        target_grid = create_target_grid(bbox, cellsize)
        coupling = make_lhm_coupling(prj_path, msw_dbase, times, target_grid=target_grid)
        coupling.write(
            out_dir,
            modflow6_dll=bin_dir/"msw.dll",
            metaswap_dll=bin_dir,
            metaswap_dll_dependency=bin_dir/"mf6.dll",
        )
