from __future__ import annotations

from abc import ABC, abstractmethod
from dask import array as da
from acryo.molecules import Molecules
from acryo._types import nm


class MinerBase(ABC):
    def __init__(self, order: int = 1) -> None:
        self._order = order

    @property
    def order(self) -> int:
        """Interpolation order."""
        return self._order

    @abstractmethod
    def find_molecules(self, image: da.Array, scale: nm = 1.0) -> Molecules:
        """Find molecules in the image of given scale."""
