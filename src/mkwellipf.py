# %%
"""MKWELLIPF script for spreading the pumping rate of wells over the model layers."""
from pathlib import Path

import imod
import numpy as np
import pandas as pd
import xarray as xr
from imod.prepare.wells import assign_wells
from imod.util.expand_repetitions import average_timeseries, resample_timeseries

from src.log import imod_logger
from src.settings import MkWellIpfSettings


def read_layers(
    files: list[Path | float], like: xr.DataArray | None = None
) -> xr.DataArray:
    """
    Stack the IDF files of the layers into one array with dimensions (layer, y, x). A number
    instead of a file is a constant for that layer, on the grid of `like`.
    """
    layers = [
        imod.idf.open(f).drop_vars("layer", errors="ignore")
        if isinstance(f, Path)
        else xr.full_like(like.isel(layer=0, drop=True), f)
        for f in files
    ]
    return xr.concat(layers, dim=pd.RangeIndex(1, len(layers) + 1, name="layer"))


def well_rates(
    wells: pd.DataFrame, ipf_file: Path, assoc_col: int, ext: str, q_col: int, iss: int,
    sdate: pd.Timestamp | None, edate: pd.Timestamp | None,
) -> np.ndarray:
    """
    The extraction volume of each well: the average of its .txt time series, or else the value
    in the rate column `q_col`. Records with nodata are left out of the average. Without valid
    records the average is 0, as in iMOD5.

    ISS=0: the average of the series as a whole, every record counting equally.
    ISS=1: the time average over a time window from `sdate` to `edate`. A record holds until the
    next one and the last record holds until `edate`. Before the first record the extraction
    volume is 0, which counts in the average.
    """
    if assoc_col == 0:
        return pd.to_numeric(wells.iloc[:, q_col - 1]).to_numpy()
    # The imod-python functions need these columns, they have no meaning for the average
    location = {"x": 0.0, "y": 0.0, "id": 0, "filt_top": 0.0, "filt_bot": 0.0, "index": 0}
    rates = []
    for name in wells.iloc[:, assoc_col - 1]:
        series = imod.ipf.read_associated(ipf_file.parent / f"{name}.{ext}")
        if series.empty:
            rates.append(0.0)
            continue
        well_rate = pd.DataFrame(
            {"time": series.iloc[:, 0], "rate": series.iloc[:, 1].astype(float)}
        ).assign(**location)
        if iss == 0:
            average = average_timeseries(well_rate)
        else:
            average = resample_timeseries(well_rate, pd.DatetimeIndex([sdate, edate]), skipna=True)
        rates.append(float(average["rate"].iloc[0]))
    return np.array(rates)


def _sample(grid: xr.DataArray, points: pd.DataFrame) -> np.ndarray:
    """Values of a (layer, y, x) grid at the points, shape (npoint, nlayer). NaN outside the grid."""
    values = np.full((len(points), grid.sizes["layer"]), np.nan)
    at = imod.select.points_values(
        grid, x=points["x"].to_numpy(), y=points["y"].to_numpy(), out_of_bounds="ignore"
    )
    values[at["index"].to_numpy()] = at.transpose("index", "layer").to_numpy()
    return values


def assign_to_layers(
    points: pd.DataFrame, top: xr.DataArray, bottom: xr.DataArray, k: xr.DataArray | None,
    minkht: float, minkd: float, fnodata: float,
) -> pd.DataFrame:
    """
    Split each well over the layers. `points` has the columns id (0, 1, ...), x, y, top, bottom
    (the screen) and rate (1.0). Returns one row per well and layer, in that order, with the
    columns FRACTION, ILAY, TOP, BOT and KD (the hydraulic conductivity). Layers are assumed to
    touch, as in MODFLOW 6.
    """
    nlayer = top.sizes["layer"]
    top_at, bottom_at = _sample(top, points), _sample(bottom, points)
    upper, lower = top_at[:, 1:], bottom_at[:, :-1]
    if (np.isfinite(upper) & np.isfinite(lower) & ~np.isclose(upper, lower)).any():
        imod_logger.warning(
            "The top of a layer differs from the bottom of the layer above it at some wells. "
            "MKWELLIPF assumes layers that touch, as in MODFLOW 6, so wells in between get no layer."
        )
    k_at = _sample(k, points) if k is not None else np.ones_like(top_at)
    fractions = np.zeros((len(points), nlayer))

    screened = points[["top", "bottom"]].replace(fnodata, np.nan).notna().all(axis=1).to_numpy()
    if screened.any():
        # Layers with K * thickness below MINKHT get no rate
        if k is not None and minkht > 0:
            k = k.where(k * (top - bottom) >= minkht)
        placed = assign_wells(points[screened], top, bottom, k, validate=False)
        fractions[placed["id"].to_numpy(), placed["layer"].to_numpy() - 1] = placed["rate"]
    if minkd > 0:
        # Without a (valid) screen, the first layer with a transmissivity above MINKD gets the well
        enough = k_at * (top_at - bottom_at) > minkd
        use = ~screened & enough.any(axis=1)
        fractions[use, enough.argmax(axis=1)[use]] = 1.0

    return pd.DataFrame(
        {
            "FRACTION": fractions.ravel(),
            "ILAY": np.tile(np.arange(1, nlayer + 1), len(points)),
            "TOP": top_at.ravel(),
            "BOT": bottom_at.ravel(),
            "KD": k_at.ravel(),
        }
    )


def build_results(
    wells: pd.DataFrame, rates: np.ndarray, layers: pd.DataFrame | None, q_col: int, hnodata: float
) -> dict[str, pd.DataFrame]:
    """
    The tables that iMOD5 writes, by file name: the input columns with column `q_col` replaced
    by Q_ASSIGNED, followed by Q_ORG (the replaced values) and, with layers, FRACTION, ILAY, TOP,
    BOT, KD and ERROR_CODE ("#" for a well without any layer, "-" otherwise).
    """
    q_name = wells.columns[q_col - 1]
    if layers is None:
        rows = wells.assign(Q_ORG=wells[q_name])
        rows[q_name] = rates
        return {"imod_mkipf_wells": rows.rename(columns={q_name: "Q_ASSIGNED"})}

    nlayer = len(layers) // len(wells)
    rows = wells.loc[wells.index.repeat(nlayer)].reset_index(drop=True)
    rows = rows.assign(Q_ORG=rows[q_name])
    fraction = layers["FRACTION"].to_numpy()
    rows[q_name] = np.where(fraction > 0, fraction * np.repeat(rates, nlayer), hnodata)
    rows = pd.concat([rows.rename(columns={q_name: "Q_ASSIGNED"}), layers], axis=1)
    assigned = fraction.reshape(-1, nlayer).sum(axis=1) > 0
    rows["ERROR_CODE"] = np.repeat(np.where(assigned, "-", "#"), nlayer)

    tables = {
        "imod_mkipf_wells_all": rows,
        "imod_mkipf_wells_unassigned": rows[rows["ERROR_CODE"] == "#"],
    }
    for layer, table in rows[rows["Q_ASSIGNED"] != hnodata].groupby("ILAY"):
        tables[f"imod_mkipf_wells_l{layer}"] = table
    return tables


def mkwellipf(settings: MkWellIpfSettings) -> list[Path]:
    """
    Run MKWELLIPF as described by the settings, see `MkWellIpfSettings`. Returns the files
    written, in a folder named after each IPF (without extension).
    """
    sdate, edate = (
        pd.Timestamp(d) if d is not None else None for d in (settings.sdate, settings.edate)
    )

    top = bottom = k = None
    if settings.top_files is not None:
        top, bottom = read_layers(settings.top_files), read_layers(settings.bottom_files)
        if settings.khv_files is not None:
            k = read_layers(settings.khv_files, like=top)
        if settings.kd_files is not None:
            k = read_layers(settings.kd_files, like=top) / (top - bottom).where(top > bottom)
    # iMOD5 only uses MINKD together with KDVIDF
    minkd = settings.minkd if settings.kd_files is not None else 0.0

    written = []
    for ipf_file in settings.ipf_files:
        imod_logger.info(f"MKWELLIPF: processing {ipf_file}")
        # Private, but imod-python's project file reader uses it too. Unlike imod.ipf.read it also
        # gives the column with the .txt file names and their extension. dtype=str keeps values
        # like well name 007 as they are.
        wells, assoc_col, ext = imod.formats.ipf._read_ipf(
            ipf_file, {"dtype": str, "keep_default_na": False}
        )
        if settings.iss == 1 and assoc_col == 0:
            raise ValueError(f"ISS=1 needs .txt files containing time-series data, but {ipf_file} has none.")
        rates = well_rates(wells, ipf_file, assoc_col, ext, settings.q_col, settings.iss, sdate, edate)

        layers = None
        if top is not None:
            number = lambda col: pd.to_numeric(wells.iloc[:, col - 1], errors="coerce")  # noqa: E731
            points = pd.DataFrame(
                {"id": np.arange(len(wells)), "x": number(settings.x_col), "y": number(settings.y_col),
                 "top": number(settings.top_col), "bottom": number(settings.bottom_col), "rate": 1.0}
            )
            layers = assign_to_layers(points, top, bottom, k, settings.minkht, minkd, settings.fnodata)

        # With ISS=1 the column with the file names is replaced, Q_ORG then holds the names
        replaced_col = assoc_col if settings.iss == 1 else settings.q_col
        tables = build_results(wells, rates, layers, replaced_col, settings.hnodata)
        folder = ipf_file.with_suffix("")
        paths = [folder / f"{name}.ipf" for name in tables]
        existing = [path for path in paths if path.exists()]
        if existing and not settings.overwrite:
            raise FileExistsError(f"Result file already exists: {existing[0]}, use overwrite.")

        folder.mkdir(exist_ok=True)
        index_col = assoc_col if settings.iss == 0 else wells.shape[1] + 1
        for path, table in zip(paths, tables.values()):
            table = table.round({"Q_ASSIGNED": 3, "FRACTION": 4, "TOP": 2, "BOT": 2, "KD": 5})
            imod.ipf.write(path, table, index_col, ext, nodata=settings.fnodata)
        written += paths

        if layers is not None:
            totals = tables["imod_mkipf_wells_all"].groupby("ILAY")["Q_ASSIGNED"].sum()
            imod_logger.info(f"Assigned rate per layer: {totals.round(3).to_dict()}")
    return written
