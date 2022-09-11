from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    # To avoid mypy error caused by scipy.
    # fmt: off
    def ifftn(arr: np.ndarray) -> np.ndarray: ...
    # fmt: on
else:
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
            np.real(ifftn(subvolume)),
            np.real(ifftn(template * missing_wedge)),
            upsample_factor=20,
            max_shifts=max_shifts,
        )
        return shift, self._DUMMY_QUAT, zncc
