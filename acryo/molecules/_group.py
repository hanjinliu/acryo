from __future__ import annotations
from typing import (
    Hashable,
    TYPE_CHECKING,
    Iterator,
    TypeVar,
    Generic,
)

if TYPE_CHECKING:
    from polars.dataframe.groupby import GroupBy
    from .core import Molecules

_K = TypeVar("_K", bound=Hashable)


class MoleculeGroup(Generic[_K]):
    """A groupby-like object for molecules."""

    def __init__(self, group: GroupBy):
        self._group = group

    def __iter__(self) -> Iterator[tuple[_K, Molecules]]:
        from .core import Molecules

        for key, df in self._group:
            mole = Molecules.from_dataframe(df)
            yield key, mole  # type: ignore

    @property
    def features(self) -> GroupBy:
        """Return the groupby object."""
        return self._group
