from __future__ import annotations
from typing import (
    TYPE_CHECKING,
    Iterator,
    NamedTuple,
    cast,
)

if TYPE_CHECKING:
    from polars.dataframe.groupby import GroupBy
    from .core import Molecules


class MoleculeCutGroup:
    """A groupby-like object for molecules created by :meth:`cutby`."""

    def __init__(self, group: GroupBy, label: str):
        self._group = group
        self._label = label

    def __iter__(self) -> Iterator[tuple[CutEdges, Molecules]]:
        from .core import Molecules

        for key, df in self._group:
            key = cast(str, key)
            gt, le = map(float, key[1:-1].split(", "))
            mole = Molecules.from_dataframe(df.drop(self._label))
            yield CutEdges(gt, le), mole


class CutEdges(NamedTuple):
    """Tuple of cut edges."""

    gt: float
    le: float

    def __repr__(self) -> str:
        return f"({self.gt}, {self.le}]"
