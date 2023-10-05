from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from acryo import _utils
from acryo._types import nm
from acryo._typed_scipy import shift as ndi_shift
from acryo.pipe._curry import converter_function

if TYPE_CHECKING:
    from acryo._typed_scipy import Mode


@converter_function
def center_by_mass(
    img: NDArray[np.float32], scale: nm, *, order: int = 3
) -> NDArray[np.float32]:
    """Centering an image by its center of mass."""
    shift = np.array(ndi.center_of_mass(img)) - np.array(img.shape) / 2
    return ndi_shift(img, -shift, order=order, prefilter=order > 1, mode="nearest")


@converter_function
def gaussian_filter(
    img: NDArray[np.float32], scale: nm, *, sigma: nm, mode="reflect", cval: float = 0.0
) -> NDArray[np.float32]:
    """Gaussian filtering an image."""
    return ndi.gaussian_filter(img, sigma / scale, mode=mode, cval=cval)  # type: ignore


@converter_function
def lowpass_filter(img: NDArray[np.float32], scale: nm, cutoff: float, order: int = 2):
    return _utils.lowpass_filter(img, cutoff=cutoff, order=order)


@converter_function
def highpass_filter(img: NDArray[np.float32], scale: nm, cutoff: float, order: int = 2):
    return _utils.highpass_filter(img, cutoff=cutoff, order=order)


@converter_function
def shift(
    img: NDArray[np.float32],
    scale: nm,
    shift: tuple[nm, nm, nm],
    *,
    order: int = 3,
    mode: Mode = "nearest",
    cval: float = 0.0,
) -> NDArray[np.float32]:
    """Shift the image by nm."""
    shift_px = np.asarray(shift) / scale
    return ndi_shift(img, shift_px, order=1, prefilter=order > 1, mode=mode, cval=cval)
