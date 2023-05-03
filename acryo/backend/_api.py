from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Literal, Protocol, Sequence, TypeVar, overload
import numpy as np
from numpy.typing import NDArray
from . import _bandpass, _missing_wedge

from acryo._types import degree
from scipy.spatial.transform import Rotation

_T = TypeVar("_T", bound=np.generic)
_T1 = TypeVar("_T1", bound=np.generic)

# fmt: off
class AnyArray(Protocol[_T]):
    def __add__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...
    def __sub__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...
    def __mul__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...
    def __truediv__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...
    def __gt__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...
    def __lt__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...
    def __ge__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...
    def __le__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...
    def __eq__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...
    def __getitem__(self, key) -> AnyArray[_T]: ...
    @property
    def real(self) -> AnyArray[np.float32]: ...
    @property
    def imag(self) -> AnyArray[np.float32]: ...
    def conj(self) -> AnyArray[_T]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def dtype(self) -> np.dtype[_T]: ...
    def dot(self, other: AnyArray[_T]) -> AnyArray[_T]: ...
    def astype(self, dtype: _T1) -> AnyArray[_T1]: ...
    @overload
    def mean(self, axis: None = None) -> _T: ...
    @overload
    def mean(self, axis: int | tuple[int, ...]) -> AnyArray[_T]: ...

# fmt: on


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
        return self._xp_.asnumpy(x)  # type: ignore

    def array(self, x, dtype: _T | None = None) -> AnyArray[_T]:
        return self._xp_.array(x, dtype)  # type: ignore

    def asarray(self, x, dtype=None) -> AnyArray[np.float32]:
        """Convert to numpy array."""
        return self._xp_.asarray(x, dtype)  # type: ignore

    @overload
    def arange(self, *args, dtype: _T, **kwargs) -> AnyArray[_T]:
        ...

    @overload
    def arange(self, *args, dtype: None = None, **kwargs) -> AnyArray:
        ...

    def arange(self, *args, dtype: _T | None = None, **kwargs) -> AnyArray[_T]:
        """Return evenly spaced values within a given interval."""
        return self._xp_.arange(*args, dtype=dtype, **kwargs)  # type: ignore

    def zeros(self, shape: int | tuple[int, ...], dtype: _T) -> AnyArray[_T]:
        return self._xp_.zeros(shape, dtype)  # type: ignore

    def fftn(
        self,
        x: AnyArray[np.float32] | AnyArray[np.complex64],
        s: tuple[int, int, int] | None = None,
        axes: int | tuple[int, ...] | None = None,
    ) -> AnyArray[np.complex64]:
        """N-dimensional FFT."""
        return self._fft_.fftn(x, s, axes)  # type: ignore

    def ifftn(
        self,
        x: AnyArray[np.float32] | AnyArray[np.complex64],
        s: tuple[int, int, int] | None = None,
        axes: int | tuple[int, ...] | None = None,
    ) -> AnyArray[np.complex64]:
        """N-dimensional inverse FFT."""
        return self._fft_.ifftn(x, s, axes)  # type: ignore

    def rfftn(
        self,
        x: AnyArray[np.float32],
        s: tuple[int, int, int] | None = None,
        axes: int | tuple[int, ...] | None = None,
    ) -> AnyArray[np.complex64]:
        """N-dimensional FFT of real part."""
        return self._fft_.rfftn(x, s, axes)  # type: ignore

    def irfftn(
        self,
        x: AnyArray[np.float32],
        s: tuple[int, int, int] | None = None,
        axes: int | tuple[int, ...] | None = None,
    ) -> AnyArray[np.complex64]:
        """N-dimensional inverse FFT of real part."""
        return self._fft_.irfftn(x, s, axes)  # type: ignore

    def fftshift(self, x: AnyArray[_T]) -> AnyArray[_T]:
        """Shift zero-frequency component to center."""
        return self._xp_.fft.fftshift(x)  # type: ignore

    def ifftshift(self, x: AnyArray[_T]) -> AnyArray[_T]:
        """Inverse shift zero-frequency component to center."""
        return self._xp_.fft.ifftshift(x)

    def fftfreq(self, n: int, d: float = 1.0) -> AnyArray[np.float32]:
        """Return the Discrete Fourier Transform sample frequencies."""
        return self._xp_.fft.fftfreq(n, d)

    @overload
    def meshgrid(
        self,
        x0: AnyArray[_T],
        copy: bool = True,
        sparse: bool = False,
        indexing: Literal["xy", "ij"] = "xy",
    ) -> tuple[AnyArray[_T]]:
        ...

    @overload
    def meshgrid(
        self,
        x0: AnyArray[_T],
        x1: AnyArray[_T],
        copy: bool = True,
        sparse: bool = False,
        indexing: Literal["xy", "ij"] = "xy",
    ) -> tuple[AnyArray[_T], AnyArray[_T]]:
        ...

    @overload
    def meshgrid(
        self,
        x0: AnyArray[_T],
        x1: AnyArray[_T],
        x2: AnyArray[_T],
        copy: bool = True,
        sparse: bool = False,
        indexing: Literal["xy", "ij"] = "xy",
    ) -> tuple[AnyArray[_T], AnyArray[_T], AnyArray[_T]]:
        ...

    def meshgrid(
        self,
        *xi: AnyArray[_T],
        copy: bool = True,
        sparse: bool = False,
        indexing: str = "xy",
    ) -> tuple[AnyArray[_T], ...]:
        """Return coordinate matrices from coordinate vectors."""
        return self._xp_.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)  # type: ignore

    def unravel_index(self, indices, shape: tuple[int, ...]) -> AnyArray[np.intp]:
        return self._xp_.unravel_index(indices, shape)

    def stack(self, arrays: Sequence[AnyArray[_T]], axis: int = 0) -> AnyArray[_T]:
        """Stack arrays in sequence along a new axis."""
        return self._xp_.stack(arrays, axis=axis)  # type: ignore

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
        return self._ndi_.affine_transform(
            self.asarray(img),
            self.asarray(matrix),
            output_shape=output_shape,
            output=output,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=prefilter,
        )  # type: ignore

    def spline_filter(
        self,
        input,
        order: int = 3,
        output: _T = np.float64,
        mode: str = "mirror",
    ) -> AnyArray[_T]:
        return self._ndi_.spline_filter(input, order=order, output=output, mode=mode)

    def map_coordinates(
        self,
        x: AnyArray[_T],
        coords: AnyArray[_T],
        order: int = 3,
        mode: str = "constant",
        cval: float = -1.0,
        prefilter: bool = True,
    ) -> AnyArray[_T]:
        return self._ndi_.map_coordinates(
            x, coords, order=order, mode=mode, cval=cval, prefilter=True
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

    def missing_wedge_mask(
        self,
        rotator: Rotation,
        tilt_range: tuple[degree, degree],
        shape: tuple[int, int, int],
    ):
        return _missing_wedge.missing_wedge_mask(self, rotator, tilt_range, shape)


NUMPY_BACKEND = Backend("numpy")


@contextmanager
def using(name: str):
    """Context manager to temporarily change the default backend."""
    old_backend = Backend._default
    Backend._default = name
    try:
        yield
    finally:
        Backend._default = old_backend