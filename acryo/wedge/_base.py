from __future__ import annotations

from typing import Iterable
from functools import reduce
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class MissingWedgeBase(ABC):
    @abstractmethod
    def create_mask(
        self,
        rotator: Rotation,
        shape: tuple[int, int, int],
    ) -> NDArray[np.float32]:
        """Create a missing wedge mask."""


class NoWedge(ABC):
    def create_mask(
        self,
        rotator: Rotation,
        shape: tuple[int, int, int],
    ) -> NDArray[np.float32]:
        return np.ones(shape, dtype=np.float32)


class UnionAxes(MissingWedgeBase):
    def __init__(self, wedges: Iterable[MissingWedgeBase]):
        self._wedges = list(wedges)

    def __repr__(self) -> str:
        reprs = ", ".join(repr(w) for w in self._wedges)
        return f"{self.__class__.__name__}({reprs})"

    def create_mask(
        self,
        rotator: Rotation,
        shape: tuple[int, int, int],
    ) -> NDArray[np.float32]:
        return reduce(np.maximum, (w.create_mask(rotator, shape) for w in self._wedges))
