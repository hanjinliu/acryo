from __future__ import annotations

from scipy import ndimage as ndi
from acryo.pipe._curry import provider_function
from acryo._reader import REG


@provider_function
def reader(scale: float, path: str, original_scale: float = None, tol=0.01):
    """
    An image provider function with rescaling.

    This function will provide a subtomogram loader with a resized image from a file.
    Will be used for the template images or the mask images.

    >>> loader.align(
    ...     template=reader("path/to/template.mrc"),
    ...     mask=reader("path/to/mask.mrc"),
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
