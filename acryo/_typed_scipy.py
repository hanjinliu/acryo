from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
import numpy as np
from numpy.typing import NDArray

# scipy is not well typed. Make patches here.
if TYPE_CHECKING:
    Mode = Literal["constant", "nearest", "mirror", "wrap", "reflect"]

    def fftn(
        img: NDArray[np.float32] | NDArray[np.complex64], s=None, axes=None
    ) -> NDArray[np.complex64]:
        ...

    def ifftn(img: NDArray[np.complex64], s=None, axes=None) -> NDArray[np.complex64]:
        ...

    def rfftn(img: NDArray[np.float32], s=None, axes=None) -> NDArray[np.complex64]:
        ...

    def irfftn(img: NDArray[np.complex64], s=None, axes=None) -> NDArray[np.float32]:
        ...

    def convolve(
        in1: NDArray[np.float32],
        in2: NDArray[np.float32],
        mode: Literal["full", "valid", "same"] = "full",
        method: Literal["auto", "fft", "direct"] = "auto",
    ) -> NDArray[np.float32]:
        ...

    def spline_filter(
        input: NDArray[np.float32],
        order: int = 3,
        output: Any = np.float64,
        mode: Mode = "mirror",
    ) -> NDArray[np.float32]:
        ...

    def affine_transform(
        img: NDArray[np.float32],
        matrix: NDArray[np.float32],
        offset: float = 0.0,
        output_shape: tuple[int, ...] | None = None,
        output: NDArray[np.float32] | None = None,
        order: int = 3,
        mode: Mode = "constant",
        cval: float = 0.0,
        prefilter: bool = True,
    ) -> NDArray[np.float32]:
        ...

    def shift(
        input: NDArray[np.float32],
        shift: tuple[float, ...] | NDArray[np.float32],
        output: NDArray[np.float32] | None = None,
        order: int = 3,
        mode: Mode = "constant",
        cval: float = 0.0,
        prefilter: bool = True,
    ) -> NDArray[np.float32]:
        ...

    def zoom(
        input: NDArray[np.float32],
        zoom: float | tuple[float, ...] | NDArray[np.float32],
        output: NDArray[np.float32] | None = None,
        order: int = 3,
        mode: Mode = "constant",
        cval: float = 0.0,
        prefilter: bool = True,
    ) -> NDArray[np.float32]:
        ...

    def map_coordinates(
        input: NDArray[np.float32],
        coordinates: NDArray[np.float32],
        output: NDArray[np.float32] | None = None,
        order: int = 3,
        mode: Mode = "constant",
        cval: float = 0.0,
        prefilter: bool = True,
    ) -> NDArray[np.float32]:
        ...

    def sum_labels(
        arr: NDArray[np.float32],
        labels: NDArray[np.integer],
        index: NDArray[np.integer],
    ) -> NDArray[np.float32]:
        ...

else:
    from scipy.fft import rfftn, irfftn, fftn, ifftn
    from scipy.ndimage import (
        spline_filter,
        affine_transform,
        map_coordinates,
        shift,
        zoom,
        sum_labels,
    )
    from scipy.signal import convolve

__all__ = [
    "fftn",
    "ifftn",
    "rfftn",
    "irfftn",
    "spline_filter",
    "affine_transform",
    "shift",
    "zoom",
    "sum_labels",
    "map_coordinates",
    "convolve",
]
