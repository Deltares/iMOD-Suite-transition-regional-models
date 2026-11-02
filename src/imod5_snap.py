"""
Port of the iMOD5 5.6.1 GENSNAPTOGRID algorithm.

The intersection and node attribution (:func:`snap_lines_imod5`) are scalar
Python that follows the Fortran. The output rules of GENSNAPTOGRID that are
needed to reproduce its files (selection of the walls between two cells, their
order and ends, and the walls of ``I3D=1``) are the functions after it.

The Fortran source is the specification:

- ``imod_intersect\\imod_intersect.f90``: ``INTERSECT_EQUI``, ``INTERSECT_NONEQUI``,
  ``INTERSECT_EQUATION``, ``INTERSECT_SORT`` and ``INTERSECT_NCORNER``.
- ``imod_utl\\imod_qksort.f90``: ``QKSORT_DBL``, the (unstable) quicksort that
  orders the intersection points.
- ``imod_polint\\imod_polint.f90``: ``POL1LOCATE``.
- ``imod_asc2idf\\imod_asc2idf_hfb.f90``: ``ASC2IDF_HFB`` (pairing of consecutive
  pieces, the final self-pair) and ``ASC2IDF_HFB_GETFACES`` (nodes to walls).

Every segment is cut at the grid lines and at the lines through the cell
centres. Each piece is attributed to the nearest corner node of the cell that
holds its midpoint, and the walls between the nodes of consecutive pieces are
marked. The operation order and exact comparisons of the Fortran are kept on
purpose: a differing tie is a defect to find with iMOD5.
"""

import itertools
import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
import shapely

from src.grid_helpers import GridSpec, x_nodes, y_nodes

COLUMNS = ["position", "kind", "irow", "icol"]
_NSTACK = 50
_INSERTION_SORT_LIMIT = 7


def _qksort(key: list[float], companion: list[float]) -> None:
    """Sort ``key`` ascending and move ``companion`` along, as ``QKSORT_DBL``."""
    n = len(key)
    k = [0.0, *key]
    c = [0.0, *companion]
    istack = [0] * (_NSTACK + 2)
    jstack = 0
    left = 1
    right = n
    while True:
        if right - left < _INSERTION_SORT_LIMIT:
            for j in range(left + 1, right + 1):
                a_key, a_comp = k[j], c[j]
                i = j - 1
                while i >= 1:
                    if k[i] <= a_key:
                        break
                    k[i + 1] = k[i]
                    c[i + 1] = c[i]
                    i -= 1
                k[i + 1] = a_key
                c[i + 1] = a_comp
            if jstack == 0:
                break
            right = istack[jstack]
            left = istack[jstack - 1]
            jstack -= 2
        else:
            mid = (left + right) // 2
            k[mid], k[left + 1] = k[left + 1], k[mid]
            c[mid], c[left + 1] = c[left + 1], c[mid]
            if k[left + 1] > k[right]:
                k[left + 1], k[right] = k[right], k[left + 1]
                c[left + 1], c[right] = c[right], c[left + 1]
            if k[left] > k[right]:
                k[left], k[right] = k[right], k[left]
                c[left], c[right] = c[right], c[left]
            if k[left + 1] > k[left]:
                k[left + 1], k[left] = k[left], k[left + 1]
                c[left + 1], c[left] = c[left], c[left + 1]
            i = left + 1
            j = right
            a_key, a_comp = k[left], c[left]
            while True:
                i += 1
                while k[i] < a_key:
                    i += 1
                j -= 1
                while k[j] > a_key:
                    j -= 1
                if j < i:
                    break
                k[i], k[j] = k[j], k[i]
                c[i], c[j] = c[j], c[i]
            k[left], c[left] = k[j], c[j]
            k[j], c[j] = a_key, a_comp
            jstack += 2
            if right - i + 1 > j - left:
                istack[jstack] = right
                istack[jstack - 1] = i
                right = j - 1
            else:
                istack[jstack] = j - 1
                istack[jstack - 1] = left
                left = i
    key[:] = k[1:]
    companion[:] = c[1:]


def _sum_of_y_steps(ya: list[float]) -> float:
    total = 0.0
    for i in range(len(ya) - 1):
        total = total + (ya[i] - ya[i + 1])
    return total


def _sort_points(
    xa: list[float], ya: list[float], dx: float, dy: float, flip: bool
) -> None:
    """``INTERSECT_SORT``: order the points along the line, in place."""
    before = _sum_of_y_steps(ya)
    if abs(dx) >= abs(dy):
        _qksort(xa, ya)
    else:
        _qksort(ya, xa)
    after = _sum_of_y_steps(ya)
    if (before < 0.0 and after > 0.0) or (before > 0.0 and after < 0.0):
        flip = not flip
    if flip:
        xa.reverse()
        ya.reverse()


def _nearest_corner(
    corners: Sequence[tuple[float, float]], xc: float, yc: float, td: float
) -> int:
    """``INTERSECT_NCORNER`` over the corners NW, NE, SE, SW; 1-based id."""
    identifier = 0
    for number, (x, y) in enumerate(corners, start=1):
        d = (xc - x) * (xc - x) + (yc - y) * (yc - y)
        if d > 0.0:
            d = math.sqrt(d)
            if d < td:
                identifier = number
                td = d
        else:
            identifier = number
            td = 0.0
    return identifier


def _seed_points(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float, bool, list[float], list[float], bool, bool, float, float]:
    """``INTERSECT_EQUATION``: order the end points west to east, seed the points."""
    flip = False
    if x1 > x2:
        flip = True
        x1, y1, x2, y2 = x2, y2, x1, y1
    xa = [x1, x1, (x1 + x2) / 2.0, x2, x2]
    ya = [y1, y1, (y1 + y2) / 2.0, y2, y2]
    dx = x2 - x1
    dy = y2 - y1
    vertical = dx == 0.0
    horizontal = dy == 0.0
    a = b = 0.0
    if dx != 0.0:
        a = dy / dx
        b = y1 - a * x1
    return x1, y1, x2, y2, flip, xa, ya, vertical, horizontal, a, b


def _filter_to_bounding_box(
    xa: list[float],
    ya: list[float],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[list[float], list[float]]:
    xmn, xmx = min(x1, x2), max(x1, x2)
    ymn, ymx = min(y1, y2), max(y1, y2)
    xs: list[float] = []
    ys: list[float] = []
    for x, y in zip(xa, ya):
        if x < xmn or x > xmx or y < ymn or y > ymx:
            continue
        xs.append(x)
        ys.append(y)
    return xs, ys


def _equi_pieces(
    xin1: float, yin1: float, xin2: float, yin2: float, grid: GridSpec
) -> list[tuple[int, int, int]]:
    """
    ``INTERSECT_EQUI`` with ``LHFB=.TRUE.`` and ``LREFINE=.FALSE.``.

    Returns the pieces inside the grid as ``(icol, irow, corner)``, 1-based, in
    the order of the line.
    """
    csx_in = grid.csx
    csy_in = grid.csy
    xmin = grid.xmin - csx_in
    xmax = grid.xmin + grid.ncol * csx_in + csx_in
    ymax = grid.ymax + csy_in
    ymin = (grid.ymax - grid.nrow * csy_in) - csy_in
    csx = csx_in / 2.0
    csy = csy_in / 2.0

    x1, y1, x2, y2, flip, xa, ya, vertical, horizontal, a, b = _seed_points(
        xin1, yin1, xin2, yin2
    )

    if x1 <= xmin:
        xmn = xmin
    else:
        xmn = xmin + csx * int((x1 - xmin) / csx)
    if x2 >= xmax:
        xmx = xmax
    else:
        xmx = xmin + csx * int((x2 - xmin) / csx)

    y = min(y1, y2)
    if y <= ymin:
        ymn = ymin
    else:
        i = 0
        if math.fmod(ymax - y, csy) == 0.0:
            i = -1
        ymn = ymax - csy * (int((ymax - y) / csy) + i)
    y = max(y1, y2)
    if y >= ymax:
        ymx = y
    else:
        ymx = ymax - csy * (int((ymax - y) / csy) + 1)

    if not horizontal:
        y = ymn
        if math.fmod(ymn, csy) == 0.0:
            y = ymn - csy
        while True:
            y = y + csy
            if y > ymx:
                break
            xa.append(x1 if vertical else (y - b) / a)
            ya.append(y)

    if not vertical:
        x = xmn
        if math.fmod(xmn, csx) == 0.0:
            x = xmn - csx
        while True:
            x = x + csx
            if x > xmx:
                break
            xa.append(x)
            ya.append(y1 if horizontal else a * x + b)

    xa, ya = _filter_to_bounding_box(xa, ya, x1, y1, x2, y2)
    _sort_points(xa, ya, x1 - x2, y2 - y1, flip)

    xmn_line, xmx_line = min(x1, x2), max(x1, x2)
    ymn_line, ymx_line = min(y1, y2), max(y1, y2)
    diagonal = math.sqrt(csx_in * csx_in + csy_in * csy_in)
    pieces: list[tuple[int, int, int]] = []
    for i in range(1, len(xa)):
        x = (xa[i - 1] + xa[i]) / 2.0
        y = (ya[i - 1] + ya[i]) / 2.0
        if x < xmn_line or x > xmx_line or y < ymn_line or y > ymx_line:
            continue
        icol = 0
        if (xmn_line < x < xmx_line) or x1 == x2:
            icol = int((x - xmin) / csx_in) + 1
        irow = 0
        if (ymn_line < y < ymx_line) or y1 == y2:
            irow = int((ymax - y) / csy_in) + 1
        # The grid was extended by one cell on every side.
        if not (2 <= icol <= grid.ncol + 1 and 2 <= irow <= grid.nrow + 1):
            continue
        corners = (
            (xmin + (icol - 1) * csx_in, ymax - (irow - 1) * csy_in),
            (xmin + icol * csx_in, ymax - (irow - 1) * csy_in),
            (xmin + icol * csx_in, ymax - irow * csy_in),
            (xmin + (icol - 1) * csx_in, ymax - irow * csy_in),
        )
        pieces.append((icol - 1, irow - 1, _nearest_corner(corners, x, y, diagonal)))
    return pieces


def _locate(nodes: Sequence[float], x: float) -> int:
    """``POL1LOCATE`` on ``nodes`` (1-based in the Fortran)."""
    n = len(nodes)
    ascending = nodes[n - 1] > nodes[0]
    jl = 0
    ju = n + 1
    while ju - jl > 1:
        jm = (ju + jl) // 2
        if ascending == (x > nodes[jm - 1]):
            jl = jm
        else:
            ju = jm
    return jl


def _nonequi_pieces(
    xin1: float, yin1: float, xin2: float, yin2: float, grid: GridSpec
) -> list[tuple[int, int, int]]:
    """
    ``INTERSECT_NONEQUI`` with ``LHFB=.TRUE.`` and ``LREFINE=.FALSE.``.

    The routine is ported as is, including the half-cell steps along x that run
    against the direction of the line and the last piece that is dropped.
    """
    delr = [float(v) for v in grid.xnodes]
    delc = [float(v) for v in grid.ynodes]
    ncol, nrow = grid.ncol, grid.nrow

    x1, y1, x2, y2, flip, xa, ya, vertical, horizontal, a, b = _seed_points(
        xin1, yin1, xin2, yin2
    )

    imn = 0
    ymn = ymx = 0.0
    for i in range(1, nrow + 1):
        if delc[i] > min(y1, y2):
            ymn = delc[i]
            imn = i
    jmn = 0
    for i in range(nrow, 0, -1):
        if delc[i] < max(y1, y2):
            ymx = delc[i]
            jmn = i

    if imn > 0 and jmn > 0 and not horizontal:
        cs = (delc[imn - 1] - delc[imn]) / 2.0
        y = ymn
        while True:
            y = y + cs
            if y > ymx:
                break
            xa.append(x1 if vertical else (y - b) / a)
            ya.append(y)
            y = y + cs
            xa.append(x1 if vertical else (y - b) / a)
            ya.append(y)
            imn -= 1
            if imn <= 0:
                break
            cs = (delc[imn] - delc[imn + 1]) / 2.0

    imn = 0
    xmn = xmx = 0.0
    for i in range(ncol, 0, -1):
        if delr[i] > x1:
            xmn = delr[i]
            imn = i
    jmn = 0
    for i in range(0, ncol + 1):
        if delr[i] < x2:
            xmx = delr[i]
            jmn = i

    if imn > 0 and jmn > 0 and not vertical:
        cs = (delr[imn - 1] - delr[imn]) / 2.0
        x = xmn
        while True:
            x = x + cs
            if x > xmx:
                break
            xa.append(x)
            ya.append(y1 if horizontal else a * x + b)
            x = x + cs
            xa.append(x)
            ya.append(y1 if horizontal else a * x + b)
            imn += 1
            if imn > ncol:
                break
            cs = (delr[imn] - delr[imn - 1]) / 2.0

    xa, ya = _filter_to_bounding_box(xa, ya, x1, y1, x2, y2)
    _sort_points(xa, ya, x1 - x2, y2 - y1, flip)

    xmn_line, xmx_line = min(x1, x2), max(x1, x2)
    ymn_line, ymx_line = min(y1, y2), max(y1, y2)
    pieces: list[tuple[int, int, int]] = []
    # N=N-2 in the Fortran: the last piece is not reported.
    for i in range(1, len(xa) - 1):
        x = (xa[i - 1] + xa[i]) / 2.0
        y = (ya[i - 1] + ya[i]) / 2.0
        if x < xmn_line or x > xmx_line or y < ymn_line or y > ymx_line:
            continue
        icol = 0
        if (xmn_line < x < xmx_line) or x1 == x2:
            icol = _locate(delr, x)
        irow = 0
        if (ymn_line < y < ymx_line) or y1 == y2:
            irow = _locate(delc, y)
        if not (1 <= icol <= ncol and 1 <= irow <= nrow):
            continue
        csx = delr[icol] - delr[icol - 1]
        csy = delc[irow - 1] - delc[irow]
        corners = (
            (delr[icol - 1], delc[irow - 1]),
            (delr[icol], delc[irow - 1]),
            (delr[icol], delc[irow]),
            (delr[icol - 1], delc[irow]),
        )
        pieces.append(
            (icol, irow, _nearest_corner(corners, x, y, math.sqrt(csx * csx + csy * csy)))
        )
    return pieces


def _faces(
    ic1: int, ic2: int, ir1: int, ir2: int, ip1: int, ip2: int
) -> list[tuple[int, int, int]]:
    """
    ``ASC2IDF_HFB_GETFACES``: the walls between the nearest corner nodes of two
    pieces, as ``(kind, ir, ic)`` 1-based; kind 0 is ``IPC(:,:,1)`` (vertical
    wall east of the cell), kind 1 is ``IPC(:,:,2)`` (horizontal wall south).
    """
    jc = (ic1, ic2)
    jr = (ir1, ir2)
    jp = (ip1, ip2)
    jpc = [0, 0]
    jpr = [0, 0]
    for i in range(2):
        if jp[i] in (2, 3):
            jpc[i] = jc[i]
        if jp[i] in (1, 4):
            jpc[i] = jc[i] - 1
        if jp[i] in (1, 2):
            jpr[i] = jr[i] - 1
        if jp[i] in (3, 4):
            jpr[i] = jr[i]

    if jpr[0] == jpr[1] and jpc[0] == jpc[1]:
        return []

    marks: list[tuple[int, int, int]] = []

    def mark(kind: int, ic: int, ir: int) -> None:
        if ic != 0 and ir != 0:
            marks.append((kind, ir, ic))

    if jpr[0] == jpr[1] and jpc[0] != jpc[1]:
        mark(1, max(jpc), jpr[0])
    if jpc[0] == jpc[1] and jpr[0] != jpr[1]:
        mark(0, jpc[0], max(jpr))
    if jpr[0] != jpr[1] and jpc[0] != jpc[1]:
        west = jpc[0] > jpc[1]
        north = jpr[0] > jpr[1]
        mark(0, min(jpc), max(jpr))
        if north:
            mark(1, max(jpc), max(jpr) if west else min(jpr))
        else:
            mark(1, max(jpc), min(jpr) if west else max(jpr))
    return marks


def _empty() -> pd.DataFrame:
    return pd.DataFrame({name: np.array([], dtype=np.int64) for name in COLUMNS})


def snap_lines_imod5(lines: Sequence[np.ndarray], grid: GridSpec) -> pd.DataFrame:
    """
    Snap lines to the walls of a grid exactly as iMOD5 GENSNAPTOGRID does.

    Parameters
    ----------
    lines : sequence of numpy.ndarray
        The features in file order, each an ``(n, 2)`` array of x, y vertices.
    grid : GridSpec
        The grid, equidistant or not.

    Returns
    -------
    pandas.DataFrame
        One row per marked wall, in the order iMOD5 marks them, without
        repeats per feature, with columns ``position`` (index in ``lines``),
        ``kind`` (0 vertical wall on the east edge of the cell, 1 horizontal wall
        on its south edge), ``irow`` and ``icol`` (0-based, row 0 is the
        northernmost). Walls on the boundary of the grid are included.
    """
    piece_function = _equi_pieces if grid.equidistant else _nonequi_pieces
    cols: list[int] = []
    rows: list[int] = []
    corners: list[int] = []
    owners: list[int] = []
    for position, line in enumerate(lines):
        vertices = np.asarray(line, dtype=np.float64).reshape(-1, 2).tolist()
        for (x1, y1), (x2, y2) in itertools.pairwise(vertices):
            for icol, irow, corner in piece_function(x1, y1, x2, y2, grid):
                cols.append(icol)
                rows.append(irow)
                corners.append(corner)
                owners.append(position)

    count = len(cols)
    result: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()

    def collect(position: int, marks: list[tuple[int, int, int]]) -> None:
        for kind, ir, ic in marks:
            row = (position, kind, ir - 1, ic - 1)
            if row not in seen:
                seen.add(row)
                result.append(row)

    for i in range(1, count):
        if owners[i - 1] != owners[i]:
            continue
        collect(
            owners[i],
            _faces(cols[i - 1], cols[i], rows[i - 1], rows[i], corners[i - 1], corners[i]),
        )
    if count > 0:
        previous = corners[count - 2] if count > 1 else corners[count - 1]
        collect(
            owners[count - 1],
            _faces(
                cols[count - 1],
                cols[count - 1],
                rows[count - 1],
                rows[count - 1],
                previous,
                corners[count - 1],
            ),
        )

    if not result:
        return _empty()
    return pd.DataFrame(result, columns=COLUMNS).astype(np.int64)


def snap_lines_to_edges(lines: Sequence[np.ndarray], grid: GridSpec) -> pd.DataFrame:
    """
    The walls between two cells that the lines snap to, sorted by ``position``.

    Walls on the boundary of the grid are not between two cells and are never
    snapped to. Columns are those of :func:`snap_lines_imod5`; ``irow`` and
    ``icol`` are 0-based, ``kind`` is 0 for the vertical wall on the east edge of
    a cell and 1 for the horizontal wall on its south edge.
    """
    walls = snap_lines_imod5(lines, grid)
    boundary = ((walls["kind"] == 0) & (walls["icol"] == grid.ncol - 1)) | (
        (walls["kind"] == 1) & (walls["irow"] == grid.nrow - 1)
    )
    return walls[~boundary].sort_values("position", kind="stable").reset_index(drop=True)


def wall_end_points(grid: GridSpec, edges: pd.DataFrame) -> np.ndarray:
    """
    Start and end ``(x, y)`` of each wall, as an array of shape ``(n, 2, 2)``.

    Vertical walls run from north to south and horizontal walls from west to
    east, as iMOD5 writes them.
    """
    x_edges = x_nodes(grid)
    y_edges = y_nodes(grid)
    kind = edges["kind"].to_numpy()
    irow = edges["irow"].to_numpy()
    icol = edges["icol"].to_numpy()
    vertical = kind == 0
    result = np.empty((len(edges), 2, 2))
    result[:, :, 0] = np.where(
        vertical[:, np.newaxis],
        x_edges[icol + 1][:, np.newaxis],
        np.column_stack([x_edges[icol], x_edges[icol + 1]]),
    )
    result[:, :, 1] = np.where(
        vertical[:, np.newaxis],
        np.column_stack([y_edges[irow], y_edges[irow + 1]]),
        y_edges[irow + 1][:, np.newaxis],
    )
    return result


def raster_order(edges: pd.DataFrame) -> np.ndarray:
    """
    Positions in ``edges`` that sort the walls as the output loop of iMOD5 does:
    by row, then column, with the vertical wall before the horizontal wall.
    """
    return np.lexsort(
        (edges["kind"].to_numpy(), edges["icol"].to_numpy(), edges["irow"].to_numpy())
    )


def unique_walls(edges: pd.DataFrame) -> pd.DataFrame:
    """
    One row per wall, in raster order. A wall snapped by several features keeps
    the first feature (``edges`` is sorted by ``position``).
    """
    unique = edges.drop_duplicates(["kind", "irow", "icol"], keep="first")
    return unique.iloc[raster_order(unique)].reset_index(drop=True)


def vertical_walls(
    grid: GridSpec, edges: pd.DataFrame, top: np.ndarray, bottom: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Turn snapped edges into vertical walls, as ``I3D=1`` of iMOD5 does.

    Parameters
    ----------
    grid : GridSpec
    edges : pandas.DataFrame
        Unique walls in raster order, see :func:`unique_walls`.
    top, bottom : numpy.ndarray
        ``(nrow, ncol)`` arrays with the TOP and BOT of the cells, NaN for
        nodata.

    Returns
    -------
    tuple of pandas.DataFrame and numpy.ndarray
        The rows of ``edges`` that give a wall, and for each a closed polygon
        with Z: the ring A-top, B-top, B-bottom, A-bottom, A-top.

    Notes
    -----
    - An edge between two cells gets ``top = max(TOP of both cells)`` and
      ``bottom = min(BOT of both cells)``. Nothing special happens if top is
      below bottom, or equal to it: zero height walls are written.
    - An edge is dropped if TOP or BOT is nodata in one of its two cells.
    - Horizontal edges run from west to east, vertical edges from north to south.
    """
    irow = edges["irow"].to_numpy()
    icol = edges["icol"].to_numpy()
    kind = edges["kind"].to_numpy()
    # A vertical wall separates (irow, icol) from its east neighbour, a
    # horizontal wall from its south neighbour.
    other_row = irow + kind
    other_col = icol + 1 - kind
    ztop = np.maximum(top[irow, icol], top[other_row, other_col])
    zbottom = np.minimum(bottom[irow, icol], bottom[other_row, other_col])

    keep = np.isfinite(ztop) & np.isfinite(zbottom)
    kept = edges[keep].reset_index(drop=True)
    if kept.empty:
        return kept, np.empty(0, dtype=object)
    first, second = wall_end_points(grid, kept).transpose(1, 0, 2)
    ztop, zbottom = ztop[keep], zbottom[keep]
    dx = second[:, 0] - first[:, 0]
    dy = second[:, 1] - first[:, 1]
    swap = np.where(np.abs(dx) > np.abs(dy), dx < 0.0, dy > 0.0)[:, np.newaxis]
    a = np.where(swap, second, first)
    b = np.where(swap, first, second)
    rings = np.stack(
        [
            np.column_stack([a, ztop]),
            np.column_stack([b, ztop]),
            np.column_stack([b, zbottom]),
            np.column_stack([a, zbottom]),
            np.column_stack([a, ztop]),
        ],
        axis=1,
    )
    return kept, shapely.polygons(shapely.linearrings(rings))
