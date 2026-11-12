
from pathlib import Path
from typing import NamedTuple

import imod
import numpy as np
import xarray as xr
from imod.util.spatial import empty_2d, spatial_reference


def open_idf_grid(path: Path | str) -> xr.DataArray:
    """
    Read an IDF without layers or times as a two-dimensional grid.

    Dimensions of length one other than ``x`` and ``y`` (such as a single
    layer) are removed; a ValueError is raised for longer dimensions.
    """
    da = imod.idf.open(path)
    extra = [dim for dim in da.dims if dim not in ("x", "y")]
    for dim in extra:
        if da.sizes[dim] != 1:
            raise ValueError(
                f"{path} holds {da.sizes[dim]} values along {dim!r}; "
                "a single grid (max one layer, max one timestep) is required."
            )
    if extra:
        da = da.squeeze(extra, drop=True)
    da = da.transpose("y", "x").load()
    da.attrs["imod5_header"] = _imod5_header(da)
    return da


def _seven_digits(value: float) -> float:
    """
    A header real as iMOD5 reads it: ``IDFREADREAL`` reads a single precision
    value, writes it as text and reads that back as a double, which leaves
    about 7 significant digits (462821.75 becomes 462821.8).
    """
    return float(f"{np.float32(value):.6e}")


def _imod5_header(da: xr.DataArray) -> dict | None:
    """The equidistant grid of an IDF as iMOD5 reads it, or None if it is not equidistant."""
    dx, xmin, _, dy, _, ymax = spatial_reference(da)
    if np.ndim(dx) != 0 or np.ndim(dy) != 0:
        return None
    return {
        "xmin": _seven_digits(xmin),
        "ymax": _seven_digits(ymax),
        "csx": _seven_digits(abs(dx)),
        "csy": _seven_digits(abs(dy)),
        # The grid this describes: a selection of other cells is not it.
        "ncol": da.sizes["x"],
        "nrow": da.sizes["y"],
        "x0": float(da["x"][0]),
        "y0": float(da["y"][0]),
    }


def _validate_window(window: tuple[float, float, float, float]) -> None:
    xmin, ymin, xmax, ymax = window
    if not np.isfinite(window).all() or xmin >= xmax or ymin >= ymax:
        raise ValueError(
            "window must contain finite xmin, ymin, xmax, ymax with "
            "xmin < xmax and ymin < ymax."
        )

def _validate_cellsize(cellsize: float) -> None:
    if not np.isfinite(cellsize) or cellsize <= 0.0:
        raise ValueError("cellsize must be finite and greater than zero.")

def _create_grid_from_like(
    like: xr.DataArray,
    window: tuple[float, float, float, float] | None,
) -> xr.DataArray:
    if like.ndim != 2 or set(like.dims) != {"x", "y"}:
        raise ValueError(
            "like must be a two-dimensional grid with x and y dimensions."
        )
    if window is not None:
        x = like["x"]
        y = like["y"]
        like = like.sel(
            x=x[(x >= window[0]) & (x <= window[2])],
            y=y[(y >= window[1]) & (y <= window[3])],
        )
    if like.sizes["x"] == 0 or like.sizes["y"] == 0:
        raise ValueError(
            "window must select at least one grid cell in each dimension."
        )
    return like

def _create_grid_from_window(window: tuple[float, float, float, float], cellsize: float) -> xr.DataArray:
    _validate_cellsize(cellsize)
    xmin, ymin, xmax, ymax = window
    # iMOD5 (UTL_IDFSNAPTOGRID_LLC) keeps the lower-left corner and drops
    # the part of the window that does not fill a whole cell.
    ncol = int((xmax - xmin) / cellsize)
    nrow = int((ymax - ymin) / cellsize)
    if ncol < 1 or nrow < 1:
        raise ValueError(
            "window must be at least one cell size wide and high; "
            f"got {window} with cellsize {cellsize}."
        )
    return empty_2d(
        dx=cellsize,
        xmin=xmin,
        xmax=xmin + ncol * cellsize,
        dy=-cellsize,
        ymin=ymin,
        ymax=ymin + nrow * cellsize,
    )

def create_target_grid(
    like: xr.DataArray | None,
    window: tuple[float, float, float, float] | None,
    cellsize: float | None,
) -> xr.DataArray:
    if window is not None:
        _validate_window(window)

    if like is not None:
        if cellsize is not None:
            raise ValueError("Cannot provide both like and cellsize.")
        return _create_grid_from_like(like, window)
    else:
        if window is None or cellsize is None:
            raise ValueError("Provide both window and cellsize together.")
        return _create_grid_from_window(window, cellsize)

class GridSpec(NamedTuple):
    """
    A structured grid as iMOD5 sees it. Row 0 is the northernmost row.

    ``xmin`` and ``ymax`` are the north-west corner. For an equidistant grid
    ``csx`` and ``csy`` are the (positive) cell sizes and ``xnodes`` and
    ``ynodes`` are None. For a non-equidistant grid ``csx`` and ``csy`` are NaN,
    ``xnodes`` holds the ``ncol + 1`` cell edges from west to east and
    ``ynodes`` the ``nrow + 1`` cell edges from north to south.
    """

    xmin: float
    ymax: float
    csx: float
    csy: float
    ncol: int
    nrow: int
    xnodes: np.ndarray | None = None
    ynodes: np.ndarray | None = None

    @property
    def equidistant(self) -> bool:
        return self.xnodes is None


def x_nodes(grid: GridSpec) -> np.ndarray:
    """The ``ncol + 1`` cell edges from west to east, as ``IDFFILLSXSY`` computes them."""
    if grid.xnodes is not None:
        return grid.xnodes
    return np.arange(grid.ncol + 1, dtype=np.float64) * grid.csx + grid.xmin


def y_nodes(grid: GridSpec) -> np.ndarray:
    """The ``nrow + 1`` cell edges from north to south."""
    if grid.ynodes is not None:
        return grid.ynodes
    return -np.arange(grid.nrow + 1, dtype=np.float64) * grid.csy + grid.ymax


def grid_spec(target: xr.DataArray) -> GridSpec:
    """
    Read the grid definition of a two-dimensional grid with ``x`` and ``y``
    coordinates, in either y order.

    A grid with a single row or column needs a ``dx`` or ``dy`` coordinate.
    Cells of an equidistant grid may differ in size between x and y.
    """
    if target.ndim != 2 or set(target.dims) != {"x", "y"}:
        raise ValueError("target must be a two-dimensional grid with x and y dimensions.")
    x = target["x"].to_numpy()
    if x.size > 1 and x[1] < x[0]:
        raise ValueError("x must increase from west to east.")
    dx, xmin, _, dy, _, ymax = spatial_reference(target)
    ncol, nrow = target.sizes["x"], target.sizes["y"]
    if np.ndim(dx) == 0 and np.ndim(dy) == 0:
        header = target.attrs.get("imod5_header")
        if (
            header is not None
            and (header["ncol"], header["nrow"]) == (ncol, nrow)
            and header["x0"] == float(target["x"][0])
            and header["y0"] == float(target["y"][0])
        ):
            return GridSpec(
                header["xmin"], header["ymax"], header["csx"], header["csy"], ncol, nrow
            )
        return GridSpec(
            float(xmin), float(ymax), abs(float(dx)), abs(float(dy)), ncol, nrow
        )

    dx_array = np.broadcast_to(np.abs(dx), (ncol,)).astype(np.float64)
    dy_array = np.broadcast_to(np.abs(dy), (nrow,)).astype(np.float64)
    y = target["y"].to_numpy()
    if y.size > 1 and y[1] > y[0]:
        dy_array = dy_array[::-1]
    return GridSpec(
        float(xmin),
        float(ymax),
        float("nan"),
        float("nan"),
        ncol,
        nrow,
        xnodes=float(xmin) + np.concatenate([[0.0], np.cumsum(dx_array)]),
        ynodes=float(ymax) - np.concatenate([[0.0], np.cumsum(dy_array)]),
    )


def _uniform_spacing(da: xr.DataArray, dim: str) -> float | None:
    """Cell size along ``dim``, or None if it is unknown or not uniform."""
    coords = np.asarray(da[dim].values, dtype=float)
    if coords.size > 1:
        spacing = np.abs(np.diff(coords))
    else:
        size = da.coords.get(f"d{dim}")
        if size is None:
            return None
        spacing = np.abs(np.atleast_1d(size.values).astype(float))
    if not np.allclose(spacing, spacing[0], rtol=1e-6):
        return None
    return float(spacing[0])


def sample_on_grid(values: xr.DataArray, target: xr.DataArray, name: str) -> xr.DataArray:
    """
    Select the cells of ``target`` from ``values``, which may be larger.

    Raises a ValueError if ``values`` is not two-dimensional, has other cells
    than the target, or lacks a cell centre of the target.
    """
    if values.ndim != 2 or set(values.dims) != {"x", "y"}:
        raise ValueError(
            f"{name} must be a two-dimensional grid with x and y dimensions."
        )
    sizes = []
    for dim in ("x", "y"):
        own = _uniform_spacing(values, dim)
        wanted = _uniform_spacing(target, dim)
        if own is not None and wanted is not None:
            if not np.isclose(own, wanted, rtol=1e-6):
                raise ValueError(
                    f"The {name} grid has cells of size {own} along {dim}, but "
                    f"the target grid has {wanted}: both must be on the same grid."
                )
            sizes.append(wanted)
    tolerance = 1e-6 * (min(sizes) if sizes else 1.0)
    x, y = target["x"].values, target["y"].values
    try:
        selected = values.sel(x=x, y=y, method="nearest", tolerance=tolerance)
    except KeyError as e:
        raise ValueError(
            f"The {name} grid does not contain every cell centre of the target "
            "grid: both must be on the same grid."
        ) from e
    return selected.assign_coords(x=x, y=y).transpose(*target.dims)


def cell_values(values: xr.DataArray) -> np.ndarray:
    """Values as an ``(nrow, ncol)`` float array, row 0 northernmost."""
    ordered = values.sortby("y", ascending=False).transpose("y", "x")
    return np.asarray(ordered.to_numpy(), dtype=np.float64)
