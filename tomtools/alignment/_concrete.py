import numpy as np
import impy as ip
from ._base import FourierLowpassInput, RealLowpassInput


class PCCAlignment(FourierLowpassInput):
    """Alignment model using phase cross correlation."""

    def optimize(
        self,
        subvolume: ip.ImgArray,
        template: ip.ImgArray,
        max_shifts: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Optimize."""
        shift, zncc = ip.ft_pcc_maximum_with_corr(
            subvolume, template, upsample_factor=20, max_shifts=max_shifts
        )
        return shift, np.zeros(4), zncc


class ZNCCAlignment(RealLowpassInput):
    """Alignment model using zero-mean normalized cross correlation."""

    def optimize(
        self,
        subvolume: ip.ImgArray,
        template: ip.ImgArray,
        max_shifts: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Optimize."""
        shift, zncc = ip.zncc_maximum_with_corr(
            subvolume, template, upsample_factor=20, max_shifts=max_shifts
        )
        return shift, np.zeros(4), zncc
