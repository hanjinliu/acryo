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
        shift, pcc = subpixel_pcc(
            subvolume,
            self.mask_missing_wedge(template, quaternion),
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
        return pcc_landscape(
            subvolume,
            self.mask_missing_wedge(template, quaternion),
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
        shift, zncc = subpixel_zncc(
            np.real(ifftn(subvolume)),
            np.real(ifftn(self.mask_missing_wedge(template, quaternion))),
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
        return zncc_landscape_with_crop(
            np.real(ifftn(subvolume)),
            np.real(ifftn(self.mask_missing_wedge(template, quaternion))),
            max_shifts=max_shifts,
        )
