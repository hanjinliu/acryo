from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from acryo.pipe._curry import provider_function
from acryo._reader import REG
from acryo._types import nm


@provider_function
def from_file(scale: nm, path: str, original_scale: float = None, tol=0.01):
    """
    An image provider function with rescaling.

    This function will provide a subtomogram loader with a resized image from a file.
    Will be used for the template images or the mask images.

    >>> loader.align(
    ...     template=from_file("path/to/template.mrc"),
    ...     mask=from_file("path/to/mask.mrc"),
    ... )

    Parameters
    ----------
    path : str
        Path to the image.
    original_scale : float, optional
        If given, this value will be used as the image scale (nm/pixel) instead
        of the scale extracted from the image metadata.
    tol : float
        Tolerance of the scale difference. If the relative difference is smaller than
        this, the image will not be resized.
    """
    if original_scale is not None and original_scale <= 0:
        raise ValueError("original_scale must be positive")
    img, img_scale = REG.imread_array(path)
    if original_scale is None:
        original_scale = img_scale
    ratio = original_scale / scale
    if abs(ratio - 1) < tol:
        return img
    return ndi.zoom(img, ratio, order=3, prefilter=True, mode="reflect")


@provider_function
def from_gaussian(
    scale: nm,
    shape: tuple[nm, nm, nm],
    sigma: nm | tuple[nm, nm, nm] = 1.0,
    shift: tuple[nm, nm, nm] = (0.0, 0.0, 0.0),
):
    """
    An image provider function by a Gaussian function.

    This function will provide a Gaussian particle with given shape, sigma and shift from
    the center.

    >>> loader.align(
    ...     template=from_gaussian(shape=(4.8, 4.8, 4.8), sigma=1.2),
    ...     mask=from_file("path/to/mask.mrc"),
    ... )

    Parameters
    ----------
    shape : float or tuple of float
        Shape of the output image in nm.
    sigma : float or tuple of float
        Standard deviation of the Gaussian particle in nm.
    shift : tuple of float, optional
        Shift of the Gaussian particle from the center in nm.
    """
    if np.isscalar(sigma):
        sigma = (sigma,) * 3
    if np.isscalar(shape):
        shape = (shape,) * 3
    sigma_px = np.array(sigma) / scale
    shape_subpix = np.array(shape) / scale
    center_subpix = shape_subpix + np.array(shift) / scale
    shape_px = tuple(np.round(shape_subpix).astype(np.int32))

    crds = np.indices(shape_px, dtype=np.float32)

    return np.exp(
        -0.5
        * sum((xx - c) / sg for xx, c, sg in zip(crds, center_subpix, sigma_px)) ** 2
    )


@provider_function
def from_array(
    scale: float, img: np.ndarray, original_scale: float = 1.0, tol: float = 0.01
):
    """
    An image provider function using existing image array.

    This function will provide a subtomogram loader with a resized image from an array.
    Will be used for the template images or the mask images.

    >>> loader.align(
    ...     template=from_array(img, original_scale=0.28),
    ...     mask=from_file("path/to/mask.mrc"),
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
