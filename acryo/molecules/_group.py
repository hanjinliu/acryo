from __future__ import annotations
from typing import (
    Hashable,
    TYPE_CHECKING,
    Iterator,
    TypeVar,
    Generic,
)

if TYPE_CHECKING:
    from polars.dataframe.group_by import GroupBy
    from acryo.molecules.core import Molecules

_K = TypeVar("_K", bound=Hashable)


class MoleculeGroup(Generic[_K]):
    """A groupby-like object for molecules."""

    def __init__(self, group: GroupBy, single: bool = False):
        self._group = group
        self._single = single

    def __iter__(self) -> Iterator[tuple[_K, Molecules]]:
        from .core import Molecules

        for key, df in self._group:
            mole = Molecules.from_dataframe(df)
            if self._single:
                key = key[0]  # type: ignore
            yield key, mole  # type: ignore

    @property
    def features(self) -> GroupBy:
        """Return the groupby object."""
        return self._group
