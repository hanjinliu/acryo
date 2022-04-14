import numpy as np
import impy as ip
from ._base import FourierLowpassInput, RealLowpassInput
from ._utils import subpixel_pcc, subpixel_zncc


class PCCAlignment(FourierLowpassInput):
    """Alignment model using phase cross correlation."""

    def optimize(
        self,
        subvolume: ip.ImgArray,
        template: ip.ImgArray,
        max_shifts: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Optimize."""
        shift, pcc = subpixel_pcc(
            subvolume, template, upsample_factor=20, max_shifts=max_shifts
        )
        return shift, np.zeros(4), pcc


class ZNCCAlignment(RealLowpassInput):
    """Alignment model using zero-mean normalized cross correlation."""

    def optimize(
        self,
        subvolume: ip.ImgArray,
        template: ip.ImgArray,
        max_shifts: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Optimize."""
        shift, zncc = subpixel_zncc(
            subvolume, template, upsample_factor=20, max_shifts=max_shifts
        )
        return shift, np.zeros(4), zncc
