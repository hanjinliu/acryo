from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from ._curry import converter_function


def resize(img, output_scale: float, original_scale: float = 1.0, *, order: int = 3):
    """Resizing an image by a given scale factor."""
    ratio = original_scale / output_scale
    return ndi.zoom(img, ratio, order=order, prefilter=order > 1, mode="reflect")


@converter_function
def center_by_mass(img: NDArray[np.float32], *, order: int = 3) -> NDArray[np.float32]:
    """Centering an image by its center of mass."""
    shift = np.array(ndi.center_of_mass(img)) - np.array(img.shape) / 2
    return ndi.shift(img, -shift, order=order, prefilter=order > 1, mode="reflect")
