from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from acryo import _utils
from acryo.pipe._curry import converter_function


@converter_function
def center_by_mass(img: NDArray[np.float32], *, order: int = 3) -> NDArray[np.float32]:
    """Centering an image by its center of mass."""
    shift = np.array(ndi.center_of_mass(img)) - np.array(img.shape) / 2
    return ndi.shift(img, -shift, order=order, prefilter=order > 1, mode="reflect")


@converter_function
def gaussian_filter(
    img: NDArray[np.float32], sigma: float, mode="reflect", cval: float = 0.0
) -> NDArray[np.float32]:
    """Gaussian filtering an image."""
    return ndi.gaussian_filter(img, sigma, mode=mode, cval=cval)


lowpass_filter = converter_function(_utils.lowpass_filter)
highpass_filter = converter_function(_utils.highpass_filter)
