from pathlib import Path

import imod
import numpy as np
import pytest
import xarray as xr

from src import convert
from src.settings import ConvertSettings


def make_simulation(name: str) -> imod.mf6.Modflow6Simulation:
    layer = np.array([1])
    idomain = xr.DataArray(
        np.ones((1, 3, 3), dtype=int),
        coords={"layer": layer, "y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5], "dx": 1.0, "dy": -1.0},
        dims=("layer", "y", "x"),
    )
    gwf = imod.mf6.GroundwaterFlowModel()
    gwf["dis"] = imod.mf6.StructuredDiscretization(
        top=1.0, bottom=xr.full_like(idomain, 0.0, dtype=float), idomain=idomain
    )
    gwf["npf"] = imod.mf6.NodePropertyFlow(
        icelltype=0, k=xr.full_like(idomain, 1.0, dtype=float)
    )
    gwf["ic"] = imod.mf6.InitialConditions(start=0.0)
    gwf["sto"] = imod.mf6.SpecificStorage(
        specific_storage=1e-5, specific_yield=0.15, transient=False, convertible=0
    )
    gwf["oc"] = imod.mf6.OutputControl(save_head="last")
    simulation = imod.mf6.Modflow6Simulation(name)
    simulation[f"{name}_model"] = gwf
    simulation["ims"] = imod.mf6.SolutionPresetSimple(modelnames=[f"{name}_model"])
    simulation.create_time_discretization(additional_times=["2000-01-01", "2000-01-02"])
    return simulation


@pytest.mark.parametrize("dump_netcdf", [True, False])
def test_convert_model_dumps_netcdf_only_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dump_netcdf: bool
) -> None:
    prjfile = tmp_path / "model.prj"
    prjfile.write_text("")
    out_dir = tmp_path / "out"
    settings = ConvertSettings(
        prjfile_in=prjfile,
        output_folder=out_dir,
        iss="steady-state",
        dump_netcdf=dump_netcdf,
    )
    monkeypatch.setattr(convert, "open_projectfile_data", lambda _: ({}, {}))
    monkeypatch.setattr(
        convert, "make_simulation", lambda *args, **kwargs: make_simulation("imported")
    )

    convert.convert_model(settings)

    netcdf_dir = tmp_path / "out_netcdf"
    assert (out_dir / "modflow6" / "mfsim.nam").exists()
    if not dump_netcdf:
        assert not netcdf_dir.exists()
        return

    dumped = list((netcdf_dir / "modflow6").rglob("*.nc"))
    assert dumped
    reloaded = imod.mf6.Modflow6Simulation.from_file(
        netcdf_dir / "modflow6" / "imported.toml"
    )
    assert "imported_model" in reloaded
