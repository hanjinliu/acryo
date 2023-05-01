from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._base import TomographyInput
from acryo._correlation import (
    subpixel_pcc,
    subpixel_zncc,
    zncc_landscape_with_crop,
    pcc_landscape,
)
from acryo._typed_scipy import ifftn
from acryo._types import pixel


class PCCAlignment(TomographyInput):
    """Alignment model using phase cross correlation."""

    def _optimize(
        self,
        subvolume: NDArray[np.complex64],
        template: NDArray[np.complex64],
        max_shifts: tuple[pixel, pixel, pixel],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """Optimize."""
        mw = self._get_missing_wedge_mask(quaternion)
        shift, pcc = subpixel_pcc(
            subvolume * mw,
            template * mw,
            upsample_factor=20,
            max_shifts=max_shifts,
        )
        return shift, self._DUMMY_QUAT, pcc

    def _landscape(
        self,
        subvolume: NDArray[np.complex64],
        template: NDArray[np.complex64],
        max_shifts: tuple[float, float, float],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute landscape."""
        mw = self._get_missing_wedge_mask(quaternion)
        return pcc_landscape(
            subvolume * mw,
            template * mw,
            max_shifts=max_shifts,
        )


class ZNCCAlignment(TomographyInput):
    """Alignment model using zero-mean normalized cross correlation."""

    def _optimize(
        self,
        subvolume: NDArray[np.complex64],
        template: NDArray[np.complex64],
        max_shifts: tuple[pixel, pixel, pixel],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """Optimize."""
        mw = self._get_missing_wedge_mask(quaternion)
        shift, zncc = subpixel_zncc(
            np.real(ifftn(subvolume * mw)),
            np.real(ifftn(template * mw)),
            upsample_factor=20,
            max_shifts=max_shifts,
        )
        return shift, self._DUMMY_QUAT, zncc

    def _landscape(
        self,
        subvolume: NDArray[np.complex64],
        template: NDArray[np.complex64],
        max_shifts: tuple[float, float, float],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute landscape."""
        mw = self._get_missing_wedge_mask(quaternion)
        return zncc_landscape_with_crop(
            np.real(ifftn(subvolume * mw)),
            np.real(ifftn(template * mw)),
            max_shifts=max_shifts,
        )
