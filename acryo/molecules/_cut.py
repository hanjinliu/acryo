from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, NamedTuple, Sequence

if TYPE_CHECKING:
    from polars.dataframe.group_by import GroupBy
    from acryo.molecules.core import Molecules


class MoleculeCutGroup:
    """A groupby-like object for molecules created by :meth:`cutby`.

    Molecules must be grouped by the bin index, that is, the i-th group corresponds to
    the right-closed interval ``(bins[i - 1], bins[i]]``.
    """

    def __init__(self, group: GroupBy, label: str, bins: Sequence[float]):
        self._group = group
        self._label = label
        self._edges = [-float("inf"), *(float(b) for b in bins), float("inf")]

    def __iter__(self) -> Iterator[tuple[CutEdges, Molecules]]:
        from .core import Molecules

        for keys, df in self._group:
            i: int = keys[0]  # type: ignore
            mole = Molecules.from_dataframe(df.drop(self._label))
            yield CutEdges(self._edges[i], self._edges[i + 1]), mole


class CutEdges(NamedTuple):
    """Tuple of cut edges."""

    gt: float
    le: float

    def __repr__(self) -> str:
        return f"({self.gt}, {self.le}]"
