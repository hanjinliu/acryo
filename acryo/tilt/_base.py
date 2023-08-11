from __future__ import annotations

from typing import Iterable
from functools import reduce
from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class TiltSeriesModel(ABC):
    @abstractmethod
    def create_mask(
        self,
        rotator: Rotation,
        shape: tuple[int, int, int],
    ) -> NDArray[np.float32]:
        """Create a missing wedge mask."""


class NoWedge(TiltSeriesModel):
    def create_mask(
        self,
        rotator: Rotation = Rotation.identity(),
        shape: tuple[int, int, int] = (49, 49, 49),
    ) -> NDArray[np.float32]:
        """
        Create an array filled with 1.0.

        Parameters
        ----------
        rotator : Rotation
            Does not affect the result.
        shape : tuple of int
            The shape of the mask.

        Returns
        -------
        np.ndarray
            Missing wedge mask.
        """
        return np.ones(shape, dtype=np.float32)


class UnionAxes(TiltSeriesModel):
    def __init__(self, wedges: Iterable[TiltSeriesModel]):
        self._wedges = list(wedges)

    def __repr__(self) -> str:
        reprs = ", ".join(repr(w) for w in self._wedges)
        return f"{self.__class__.__name__}({reprs})"

    def create_mask(
        self,
        rotator: Rotation = Rotation.identity(),
        shape: tuple[int, int, int] = (49, 49, 49),
    ) -> NDArray[np.float32]:
        """
        Create a binary mask that covers tomographical missing wedge.

        Note that the mask is not shifted to the center of the Fourier domain.
        ``np.fft.fftn(img) * mask`` will be the correct way to apply the mask.

        Parameters
        ----------
        rotator : Rotation
            The rotation object that describes the direction of the mask.
        shape : tuple of int
            The shape of the mask.

        Returns
        -------
        np.ndarray
            Missing wedge mask.
        """
        return reduce(np.maximum, (w.create_mask(rotator, shape) for w in self._wedges))
