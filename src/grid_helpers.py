
import numpy as np
import xarray as xr
from imod.util.spatial import empty_2d


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

def create_target_grid(
    like: xr.DataArray | None,
    window: tuple[float, float, float, float] | None,
    cellsize: float | None,
) -> xr.DataArray:
    if window is not None:
        _validate_window(window)
        xmin, ymin, xmax, ymax = window

    if like is not None:
        if cellsize is not None:
            raise ValueError("Cannot provide both like and cellsize.")
        return _create_grid_from_like(like, window)
    else:
        if window is None or cellsize is None:
            raise ValueError("Provide both window and cellsize together.")
        _validate_cellsize(cellsize)
        return empty_2d(
            dx=cellsize,
            xmin=xmin,
            xmax=xmax,
            dy=-cellsize,
            ymin=ymin,
            ymax=ymax,
        )
