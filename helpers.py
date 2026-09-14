"""
Helpers to clip point-based packages to a regional domain's
bounding box *before* mask_all_models is called.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from imod.mf6.model import Modflow6Model
from imod.mf6.wel import Well
from imod.util.spatial import spatial_reference

logger = logging.getLogger(__name__)


def domain_bounds(domain) -> tuple[float, float, float, float]:
    """Get (x_min, x_max, y_min, y_max) cell-edge bounds for a domain."""
    _, x_min, x_max, _, y_min, y_max = spatial_reference(domain)
    return x_min, x_max, y_min, y_max


def clip_point_package_to_bounds(
    package: Well,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    pkgname: str,
) -> Well:
    """Drop points from a point-based package (e.g. Well) that fall outside the given bounding box."""
    # Any package exposing .clip_box(x_min=..., x_max=..., y_min=..., y_max=...)
    # with point data on an "index" dimension works here, not just Well.

    n_before = package.dataset.sizes["index"]
    clipped = package.clip_box(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
    n_after = clipped.dataset.sizes["index"]

    n_dropped = n_before - n_after
    if n_dropped > 0:
        logger.warning(
            f"Dropped {n_dropped}/{n_before} point(s) from package {pkgname} outside domain bounds "
            f"(x: {x_min:.2f}-{x_max:.2f}, y: {y_min:.2f}-{y_max:.2f})"
        )

    return clipped


def clip_wel_packages_to_bounds(
    model: Modflow6Model,
    wel_keys: Iterable[str],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Clip every wel-* package in `model` to the given bounding box."""
    for pkgname in wel_keys:
        model[pkgname] = clip_point_package_to_bounds(
            model[pkgname], x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, pkgname=pkgname
        )


def clip_wel_packages_to_domain(model: Modflow6Model, wel_keys: Iterable[str]) -> None:
    """Derive bounds from model.domain itself and clip every wel-* package to them. In-place."""
    x_min, x_max, y_min, y_max = domain_bounds(model.domain)
    clip_wel_packages_to_bounds(model, wel_keys=wel_keys, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)
