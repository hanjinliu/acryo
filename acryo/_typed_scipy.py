from __future__ import annotations

from typing import TYPE_CHECKING, Any
import numpy as np
from numpy.typing import NDArray

# scipy is not well typed. Make patches here.
if TYPE_CHECKING:

    def fftn(
        img: NDArray[np.float32] | NDArray[np.complex64], axes=None
    ) -> NDArray[np.complex64]:
        ...

    def ifftn(img: NDArray[np.complex64], axes=None) -> NDArray[np.complex64]:
        ...

    def rfftn(img: NDArray[np.float32]) -> NDArray[np.complex64]:
        ...

    def irfftn(img: NDArray[np.complex64]) -> NDArray[np.float32]:
        ...

    def spline_filter(
        input: NDArray[np.float32],
        order: int = 3,
        output: Any = np.float64,
        mode: str = "mirror",
    ) -> NDArray[np.float32]:
        ...

    def affine_transform(
        img: NDArray[np.float32],
        matrix: NDArray[np.float32],
        offset: float = 0.0,
        output_shape: tuple[int, ...] | None = None,
        output: NDArray[np.float32] | None = None,
        order: int = 3,
        mode: str = "constant",
        cval: float = 0.0,
        prefilter: bool = True,
    ) -> NDArray[np.float32]:
        ...

    def map_coordinates(
        input: NDArray[np.float32],
        coordinates: NDArray[np.float32],
        output: NDArray[np.float32] | None = None,
        order: int = 3,
        mode: str = "constant",
        cval: float = 0.0,
        prefilter: bool = True,
    ) -> NDArray[np.float32]:
        ...

else:
    from scipy.fft import rfftn, irfftn, fftn, ifftn
    from scipy.ndimage import spline_filter, affine_transform, map_coordinates

__all__ = [
    "fftn",
    "ifftn",
    "rfftn",
    "irfftn",
    "spline_filter",
    "affine_transform",
    "map_coordinates",
]
