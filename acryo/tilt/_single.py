from __future__ import annotations
from abc import abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from acryo.tilt import _base, _utils


class SingleAxis(_base.TiltSeriesModel):
    def __init__(self, tilt_range=(-60.0, 60.0)):
        _min, _max = tilt_range
        if _min >= _max:
            raise ValueError(f"Tilt range {tilt_range!r} does not satisfy min < max.")
        if _min < -90.0 or _max > 90.0:
            raise ValueError(f"Tilt range {tilt_range!r} is not between -90 and 90.")
        self._tilt_range = tilt_range

    @property
    def tilt_range(self) -> tuple[float, float]:
        """Range of tilt angles in degrees."""
        return self._tilt_range

    @abstractmethod
    def _get_norms(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Return the two normal vectors that define the missing wedge."""

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
        normal0, normal1 = self._get_norms()
        shape_vector = np.array(shape, dtype=np.float32)
        rotator_inv = rotator.inv()
        normal0 = rotator_inv.apply(normal0 * shape_vector)
        normal1 = rotator_inv.apply(normal1 * shape_vector)
        vectors = _utils.get_indices(shape)
        dot0 = vectors.dot(normal0)
        dot1 = vectors.dot(normal1)
        missing = dot0 * dot1 <= 0
        return missing

    def _mask_from_norms(
        self,
        normal0: NDArray[np.float32],
        normal1: NDArray[np.float32],
        rotator: Rotation,
        shape: tuple[int, int, int],
    ) -> NDArray[np.float32]:
        shape_vector = np.array(shape, dtype=np.float32)
        rotator_inv = rotator.inv()
        normal0 = rotator_inv.apply(normal0 * shape_vector)
        normal1 = rotator_inv.apply(normal1 * shape_vector)
        vectors = _utils.get_indices(shape)
        dot0 = vectors.dot(normal0)
        dot1 = vectors.dot(normal1)
        missing = dot0 * dot1 <= 0
        return missing


class SingleAxisY(SingleAxis):
    def _get_norms(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return _utils.get_norms_y(self._tilt_range)

    def __repr__(self) -> str:
        tilt = f"{self.tilt_range[0]:.1f}, {self.tilt_range[1]:.1f}"
        return f"SingleAxis<y>({tilt})"


class SingleAxisX(SingleAxis):
    def _get_norms(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        return _utils.get_norms_x(self._tilt_range)

    def __repr__(self) -> str:
        tilt = f"{self.tilt_range[0]:.1f}, {self.tilt_range[1]:.1f}"
        return f"SingleAxis<x>({tilt})"
