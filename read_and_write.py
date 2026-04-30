# %%
import sys
from pathlib import Path

import imod
from imod.formats.prj.prj import open_projectfile_data
from imod.logging.config import LoggerType
from imod.logging.loglevel import LogLevel
from imod.mf6.ims import Solution
from imod.mf6.oc import OutputControl
from imod.mf6.simulation import Modflow6Simulation
import pandas as pd
from primod import MetaMod, MetaModDriverCoupling


def convert_imod5_to_mf6_sim(
    imod5_data: dict, period_data: dict, times: list
) -> Modflow6Simulation:
    simulation = Modflow6Simulation.from_imod5_data(
        imod5_data,
        period_data,
        times,
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
    prjfile_path: Path, msw_dbase: Path, times: list[pd.Timestamp]
) -> tuple[Modflow6Simulation, imod.msw.MetaSwapModel]:
    """
    Convert iMOD5 LHM model to MODFLOW 6 using imod-python
    """

    # Read iMOD5 project file data
    imod5_data, period_data = open_projectfile_data(prjfile_path)

    # Convert to MODFLOW 6 simulation and cleanup
    mf6_simulation = convert_imod5_to_mf6_sim(imod5_data, period_data, times)
    cleanup_mf6_sim(mf6_simulation)

    # Convert to MetaSwap model
    msw_model = convert_imod5_to_msw_model(
        imod5_data, mf6_simulation, times, msw_dbase
    )

    return mf6_simulation, msw_model


def make_lhm_coupling(
    prjfile_path: Path, msw_dbase: Path, times: list[pd.Timestamp]
) -> MetaMod:
    """
    Test coupling of LHM MODFLOW 6 and MetaSwap models
    """
    mf6_simulation, msw_model = import_lhm_mf6_and_msw(
        prjfile_path, msw_dbase, times
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

    # notes: M/D/Y and convert to list of datetime.
    times = pd.date_range(start="1/1/2011", end="1/7/2011", freq="D").tolist()

    #Path management
    wdir = Path(r"c:\Users\engelen\projects_wdir\imod-python\imod5_converter\NHI_sprint")

    msw_dbase = wdir / "dummy_path"
    prj_path = wdir / "BASIS7" / "IBR30_BASIS7_TA-25x25-adapted.PRJ"
    out_dir = wdir / "conversion_output"
    logfile_path = out_dir / "conversion_log.txt"
    bin_dir = wdir / "bin"

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(logfile_path, "w") as sys.stdout:
        imod.logging.configure(
            LoggerType.PYTHON,
            log_level=LogLevel.DEBUG,
            add_default_file_handler=False,
            add_default_stream_handler=True,
        )

        coupling = make_lhm_coupling(prj_path, msw_dbase, times)
        coupling.write(
            out_dir,
            modflow6_dll=bin_dir/"msw.dll",
            metaswap_dll=bin_dir,
            metaswap_dll_dependency=bin_dir/"mf6.dll",
        )
