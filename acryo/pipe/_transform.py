from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from acryo import _utils
from acryo._types import nm
from acryo.pipe._curry import provider_function, converter_function


@converter_function
def center_by_mass(
    img: NDArray[np.float32], scale: nm, *, order: int = 3
) -> NDArray[np.float32]:
    """Centering an image by its center of mass."""
    shift = np.array(ndi.center_of_mass(img)) - np.array(img.shape) / 2
    return ndi.shift(img, -shift, order=order, prefilter=order > 1, mode="reflect")


@converter_function
def gaussian_filter(
    img: NDArray[np.float32], scale: nm, *, sigma: nm, mode="reflect", cval: float = 0.0
) -> NDArray[np.float32]:
    """Gaussian filtering an image."""
    return ndi.gaussian_filter(img, sigma / scale, mode=mode, cval=cval)


@converter_function
def lowpass_filter(img: NDArray[np.float32], scale: nm, cutoff: float, order: int = 2):
    return _utils.lowpass_filter(img, cutoff=cutoff, order=order)


@converter_function
def highpass_filter(img: NDArray[np.float32], scale: nm, cutoff: float, order: int = 2):
    return _utils.highpass_filter(img, cutoff=cutoff, order=order)


@provider_function
def resize(
    scale: float, img: NDArray[np.floating], original_scale: float = 1.0, tol=0.01
):
    """
    An image provider function using existing image array.

    This function will provide a subtomogram loader with a resized image from an array.
    Will be used for the template images or the mask images.

    >>> loader.align(
    ...     template=resize(img, original_scale=0.28),
    ...     mask=reader("path/to/mask.mrc"),
    ... )

    Parameters
    ----------
    img : np.ndarray
        Input image array. Must be 3D.
    original_scale : float, optional
        If given, this value will be used as the image scale (nm/pixel) instead
        of the scale extracted from the image metadata.
    tol : float
        Tolerance of the scale difference. If the relative difference is smaller than
        this, the image will not be resized.
    """
    if original_scale is not None and original_scale <= 0:
        raise ValueError("original_scale must be positive")
    if img.ndim != 3:
        raise ValueError("img must be 3D")
    ratio = original_scale / scale
    if abs(ratio - 1) < tol:
        return img
    out = ndi.zoom(img, ratio, order=3, prefilter=True, mode="reflect")
    if not out.dtype == np.float32:
        out = out.astype(np.float32)
    return out
