from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Protocol, TypeVar
import numpy as np
from numpy.typing import NDArray
from . import _bandpass

_T = TypeVar("_T", covariant=True)


class AnyArray(Protocol[_T]):
    ...


class Backend:
    _default = "numpy"

    def __init__(self, name: str | None = None) -> None:
        if name is None:
            name = self._default
        if name == "numpy":
            from scipy import ndimage, fft

            self._xp_ = np
            self._ndi_ = ndimage
            self._fft_ = fft
        elif name == "cupy":
            import cupy
            from cupyx.scipy import ndimage, fft

            self._xp = cupy
            self._ndi_ = ndimage
            self._fft_ = fft
        else:
            raise ValueError(f"Unknown backend {name}")

    @property
    def name(self) -> str:
        return self._xp_.__name__

    def __hash__(self) -> int:
        return hash(self._xp_)

    def asnumpy(self, x) -> NDArray[np.float32]:
        """Convert to numpy array."""
        if self._xp_ is np:
            return x
        return self._xp_.asnumpy(x)

    def asarray(self, x) -> AnyArray[np.float32]:
        """Convert to numpy array."""
        return self._xp_.asarray(x)

    def affine_transform(
        self,
        img,
        matrix,
        output_shape: tuple[int, ...] | None = None,
        output=None,
        order: int = 3,
        mode: str = "constant",
        cval: float = 0.0,
        prefilter: bool = True,
    ) -> AnyArray[np.float32]:
        """Affine transform."""
        self._ndi_.affine_transform(
            self.asarray(img),
            self.asarray(matrix),
            output_shape=output_shape,
            output=output,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
        )

    def rotated_crop(
        self,
        subimg,
        mtx: NDArray[np.float32],
        shape: tuple[int, int, int],
        order: int,
        cval: float | Callable[[NDArray[np.float32]], float],
    ) -> AnyArray[np.float32]:
        if callable(cval):
            cval = cval(subimg)

        out = self.affine_transform(
            subimg,
            matrix=self.asarray(mtx),
            output_shape=shape,
            order=order,
            prefilter=order > 1,
            mode="constant",
            cval=cval,
        )
        return out

    def lowpass_filter_ft(self, img, cutoff, order) -> AnyArray[np.complex64]:
        """Lowpass filter in Fourier space."""
        return _bandpass.lowpass_filter_ft(self, self.asarray(img), cutoff, order)

    def lowpass_filter(self, img, cutoff, order) -> AnyArray[np.float32]:
        """Lowpass filter in real space."""
        return _bandpass.lowpass_filter(self, self.asarray(img), cutoff, order)


@contextmanager
def using(name: str):
    """Context manager to temporarily change the default backend."""
    old_backend = Backend._default
    Backend._default = name
    try:
        yield
    finally:
        Backend._default = old_backend
