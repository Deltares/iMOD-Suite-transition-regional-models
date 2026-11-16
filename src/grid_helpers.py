
from pathlib import Path
from typing import NamedTuple

import imod
import numpy as np
import xarray as xr
import xugrid as xu
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
    return da.transpose("y", "x").load()


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
    A structured grid. Row 0 is the northernmost row.

    ``xmin`` and ``ymax`` are the north-west corner. For an equidistant grid
    ``dx`` is the cell size along x and ``dy`` the cell size along y, which is
    negative as rows run from north to south; ``xnodes`` and ``ynodes`` are
    None. For a non-equidistant grid ``dx`` and ``dy`` are NaN, ``xnodes`` holds
    the ``ncol + 1`` cell edges from west to east and ``ynodes`` the
    ``nrow + 1`` cell edges from north to south.
    """

    xmin: float
    ymax: float
    dx: float
    dy: float
    ncol: int
    nrow: int
    xnodes: np.ndarray | None = None
    ynodes: np.ndarray | None = None

    @property
    def equidistant(self) -> bool:
        return self.xnodes is None


def x_nodes(grid: GridSpec) -> np.ndarray:
    """The ``ncol + 1`` cell edges from west to east."""
    if grid.xnodes is not None:
        return grid.xnodes
    return np.arange(grid.ncol + 1, dtype=np.float64) * grid.dx + grid.xmin


def y_nodes(grid: GridSpec) -> np.ndarray:
    """The ``nrow + 1`` cell edges from north to south."""
    if grid.ynodes is not None:
        return grid.ynodes
    return np.arange(grid.nrow + 1, dtype=np.float64) * grid.dy + grid.ymax


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
        return GridSpec(
            float(xmin), float(ymax), abs(float(dx)), -abs(float(dy)), ncol, nrow
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


def regrid_to_target(values: xr.DataArray, target: xr.DataArray, name: str) -> xr.DataArray:
    """
    Area-weighted mean of ``values`` over the cells of ``target``.

    ``values`` may have other cell sizes and extent than ``target``. Nodata
    (NaN) is ignored; cells of ``target`` without any valid value are NaN.
    """
    if values.ndim != 2 or set(values.dims) != {"x", "y"}:
        raise ValueError(
            f"{name} must be a two-dimensional grid with x and y dimensions."
        )
    regridder = xu.OverlapRegridder(source=values, target=target, method="mean")
    return regridder.regrid(values)


def cell_values(values: xr.DataArray) -> np.ndarray:
    """Values as an ``(nrow, ncol)`` float array, row 0 northernmost."""
    ordered = values.sortby("y", ascending=False).transpose("y", "x")
    return np.asarray(ordered.to_numpy(), dtype=np.float64)
