from __future__ import annotations

from typing import Sequence, TypeVar
import polars as pl

POLARS_V2 = int(pl.__version__.split(".", 1)[0]) >= 2

_E = TypeVar("_E", pl.Expr, pl.Series)


def bin_index(x: _E, breaks: Sequence[float]) -> _E:
    """Index of the right-closed bins ``(breaks[i - 1], breaks[i]]`` of the values.

    Values smaller than or equal to ``breaks[0]`` are in the 0-th bin and values larger
    than ``breaks[-1]`` are in the ``len(breaks)``-th bin. The returned data type is
    ``UInt32``.
    """
    if POLARS_V2:
        # `cut` is deprecated in polars>=2
        return x.bin_intervals(breaks, labels=False, right_closed=True)
    labels = [str(i) for i in range(len(breaks) + 1)]
    return x.cut(breaks, labels=labels).cast(pl.String).cast(pl.UInt32)
