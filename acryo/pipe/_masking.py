from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from acryo.pipe._curry import converter_function
from acryo._types import nm


@converter_function
def threshold_otsu(
    img: NDArray[np.float32], scale: nm, bins: int = 256
) -> NDArray[np.bool_]:
    """
    Pipe operation that binarize an image using Otsu's method.

    Parameters
    ----------
    bins : int, default is 256
        Number of bins to build histogram.
    """
    hist, edges = np.histogram(img.ravel(), bins=bins)
    centers: NDArray[np.float32] = (edges[:-1] + edges[1:]) / 2
    npixel0 = np.cumsum(hist)
    npixel1 = img.size - npixel0

    nonzero0 = npixel0 != 0
    nonzero1 = npixel1 != 0

    mean0 = np.zeros_like(centers)
    mean1 = np.zeros_like(centers)
    product = hist * centers
    mean0[nonzero0] = np.cumsum(product)[nonzero0] / npixel0[nonzero0]
    mean1[nonzero1] = (np.cumsum((product)[nonzero1][::-1]) / npixel1[nonzero1][::-1])[
        ::-1
    ]

    s = npixel0 * npixel1 * (mean0 - mean1) ** 2

    imax = np.argmax(s)
    thr = centers[imax]
    return img > thr


@converter_function
def dilation(img: NDArray[np.bool_], scale: nm, radius: nm) -> NDArray[np.bool_]:
    """
    Pipe operation that dilate (or erode) a binary image using a circular structure.

    Parameters
    ----------
    radius : float
        Radius of the structure element in nanometer. If negative, erosion is applied.
    """
    r = _get_radius_px(radius, scale)
    if r == 0:
        return img
    structure = _get_structure(r)
    if radius < 0:
        out = ndi.binary_erosion(img, structure=structure, border_value=False)
    elif radius > 0:
        out = ndi.binary_dilation(img, structure=structure, border_value=False)
    return out  # type: ignore


@converter_function
def closing(img: NDArray[np.bool_], scale: nm, radius: nm) -> NDArray[np.bool_]:
    """
    Pipe operation that close (or open) a binary image using a circular structure.

    Parameters
    ----------
    radius : float
        Radius of the structure element in nanometer. If negative, opening is applied.
    """
    r = _get_radius_px(radius, scale)
    if r == 0:
        return img
    structure = _get_structure(r)
    if radius < 0:
        out = ndi.binary_opening(img, structure=structure, border_value=False)
    elif radius > 0:
        out = ndi.binary_closing(img, structure=structure, border_value=False)
    return out  # type: ignore


def _get_radius_px(radius: nm, scale: nm) -> int:
    try:
        radius = float(radius)
    except ValueError:
        raise ValueError(f"radius must be a number, got {radius!r}")
    radius_px = abs(radius / scale)
    if radius_px < 1:
        return 0
    return int(np.ceil(radius_px))


def _get_structure(r: int) -> NDArray[np.int32]:
    size = 2 * r + 1
    zz, yy, xx = np.indices((size,) * 3)
    return (xx - r) ** 2 + (yy - r) ** 2 + (zz - r) ** 2 <= r**2


@converter_function
def gaussian_smooth(
    img: NDArray[np.bool_], scale: nm, sigma: nm
) -> NDArray[np.float32]:
    """
    Pipe operation that smooth a binary image using a Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian kernel.
    """
    try:
        sigma = float(sigma)
    except ValueError:
        raise ValueError(f"sigma must be a number, got {sigma!r}")
    if sigma == 0:
        return img.astype(np.float32)
    if sigma < 0:
        raise ValueError(f"sigma must be positive, got {sigma!r}")
    if img.all() or not img.any():
        # all True or all False
        return img.astype(np.float32)
    img = ~img
    dist: NDArray[np.float32] = ndi.distance_transform_edt(img)  # type: ignore
    blurred_mask = np.exp(-(dist**2) / 2 / (sigma / scale) ** 2, dtype=np.float32)
    return blurred_mask


def soft_otsu(sigma: nm = 1.0, radius: nm = 1.0, bins: int = 256):
    """
    Pipe operation of soft Otsu thresholding.

    This operation binarize an image using Otsu's method, dilate the edges and
    smooth the image using a Gaussian kernel.

    >>> from acryo.pipe import reader, soft_otsu
    >>> loader.align(
    ...     template=reader("path/to/template.mrc"),
    ...     mask=soft_otsu(2.0, 2.0),
    ... )

    Parameters
    ----------
    sigma : float, default is 1.0
        Standard deviation of the Gaussian kernel.
    radius : float, default is 1.0
        Radius of the structure element. If negative, erosion is applied.
    bins : int, default is 256
        Number of bins to build histogram.
    """
    out = gaussian_smooth(sigma) @ dilation(radius) @ threshold_otsu(bins)
    return out.with_name(f"soft_otsu({sigma=:.2f}, {radius=:.2f}, {bins=})")
