from __future__ import annotations

import numpy as np

from ._base import TomographyInput
from ._utils import subpixel_pcc, subpixel_zncc
from acryo._fft import ifftn


class PCCAlignment(TomographyInput):
    """Alignment model using phase cross correlation."""

    def optimize(
        self,
        subvolume: np.ndarray,
        template: np.ndarray,
        max_shifts: tuple[float, float, float],
        quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Optimize."""
        shift, pcc = subpixel_pcc(
            subvolume,
            self.mask_missing_wedge(template, quaternion),
            upsample_factor=20,
            max_shifts=max_shifts,
        )
        return shift, self._DUMMY_QUAT, pcc


class ZNCCAlignment(TomographyInput):
    """Alignment model using zero-mean normalized cross correlation."""

    def optimize(
        self,
        subvolume: np.ndarray,
        template: np.ndarray,
        max_shifts: tuple[float, float, float],
        quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Optimize."""
        shift, zncc = subpixel_zncc(
            np.real(ifftn(subvolume)),
            np.real(ifftn(self.mask_missing_wedge(template, quaternion))),
            upsample_factor=20,
            max_shifts=max_shifts,
        )
        return shift, self._DUMMY_QUAT, zncc
