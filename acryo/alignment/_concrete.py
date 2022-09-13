from __future__ import annotations

import numpy as np
from scipy.fft import ifftn

from ._base import TomographyInput
from ._utils import subpixel_pcc, subpixel_zncc


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
        missing_wedge = self._get_missing_wedge_mask(quaternion)
        shift, pcc = subpixel_pcc(
            subvolume,
            template * missing_wedge,
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
        missing_wedge = self._get_missing_wedge_mask(quaternion)
        shift, zncc = subpixel_zncc(
            np.real(ifftn(subvolume)),  # type: ignore
            np.real(ifftn(template * missing_wedge)),  # type: ignore
            upsample_factor=20,
            max_shifts=max_shifts,
        )
        return shift, self._DUMMY_QUAT, zncc
