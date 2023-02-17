from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi


class SoftOtsu:
    """
    Image smoothing by Otsu thresholding, dilation and Gaussian blurring.

    Parameters
    ----------
    sigma : float, default is 1.0
        Standard deviation of the Gaussian kernel.
    radius : int, default is 1
        Radius of the structuring element for dilation/erosion.
    bins : int, default is 256
        Number of bins for Otsu thresholding.
    """

    def __init__(self, sigma: float = 1.0, radius: int = 1, bins: int = 256):
        self._sigma = sigma
        self._radius = radius
        self._bins = bins
        self._structure = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(radius={self._radius}, sigma={self._sigma})"

    @property
    def structure(self) -> NDArray[np.bool_]:
        if self._structure is None:
            r = abs(self._radius)
            zz, yy, xx = np.indices((2 * r + 1, 2 * r + 1, 2 * r + 1))
            self._structure = (xx - r) ** 2 + (yy - r) ** 2 + (zz - r) ** 2 <= r**2
        return self._structure

    def __call__(self, img: NDArray[np.float32]) -> NDArray[np.float32]:
        binary = img < _threshold_otsu(img, self._bins)
        if self._radius > 0:
            mask = ndi.binary_erosion(
                binary, structure=self.structure, border_value=True
            )
        elif self._radius < 0:
            mask = ndi.binary_dilation(
                binary, structure=self.structure, border_value=True
            )
        else:
            mask = binary
        dist = ndi.distance_transform_edt(mask)
        blurred_mask = np.exp(-(dist**2) / 2 / self._sigma**2)
        return blurred_mask


def _threshold_otsu(img: NDArray[np.float32], bins: int = 256) -> float:
    hist, edges = np.histogram(img.ravel(), bins=bins)
    centers: NDArray[np.float32] = (edges[:-1] + edges[1:]) / 2
    npixel0 = np.cumsum(hist)
    npixel1 = img.size - npixel0

    nonzero0 = npixel0 != 0
    nonzero1 = npixel1 != 0

    mean0 = np.zeros_like(centers)
    mean1 = np.zeros_like(centers)
    mean0[nonzero0] = np.cumsum(hist * centers)[nonzero0] / npixel0[nonzero0]
    mean1[nonzero1] = (
        np.cumsum((hist * centers)[nonzero1][::-1]) / npixel1[nonzero1][::-1]
    )[::-1]

    s = npixel0 * npixel1 * (mean0 - mean1) ** 2

    imax = np.argmax(s)
    return centers[imax]
