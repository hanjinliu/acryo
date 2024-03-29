from __future__ import annotations

from contextlib import contextmanager
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Generic,
    Sequence,
    TypeVar,
    overload,
)
import numpy as np
from numpy.typing import NDArray
from . import _bandpass, _missing_wedge

from acryo._types import degree
from scipy.spatial.transform import Rotation

_T = TypeVar("_T", bound=np.generic)
_T1 = TypeVar("_T1", bound=np.generic)

# fmt: off
class AnyArray(Generic[_T]):
    """
    Type representing a ndarray of numpy or cupy (or any other array
    that has similar API).
    """
    def __pos__(self) -> AnyArray[_T]: ...
    def __neg__(self) -> AnyArray[_T]: ...
    def __invert__(self) -> AnyArray[_T]: ...
    def __add__(self, other: Any) -> AnyArray[_T]: ...  # type: ignore
    def __sub__(self, other: Any) -> AnyArray[_T]: ...  # type: ignore
    def __mul__(self, other: Any) -> AnyArray[_T]: ...  # type: ignore
    def __truediv__(self, other: Any) -> AnyArray[np.float_]: ...  # type: ignore
    def __radd__(self, other: Any) -> AnyArray[_T]: ...  # type: ignore
    def __rsub__(self, other: Any) -> AnyArray[_T]: ...  # type: ignore
    def __rmul__(self, other: Any) -> AnyArray[_T]: ...  # type: ignore
    def __rtruediv__(self, other: Any) -> AnyArray[np.float_]: ...  # type: ignore
    def __floordiv__(self, other: Any) -> AnyArray[np.intp]: ...  # type: ignore
    def __gt__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...  # type: ignore
    def __lt__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...  # type: ignore
    def __ge__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...  # type: ignore
    def __le__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...  # type: ignore
    def __eq__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...  # type: ignore
    def __pow__(self, other: AnyArray[_T] | float) -> AnyArray[_T]: ...  # type: ignore
    def __getitem__(self, key) -> AnyArray[_T]: ...  # type: ignore
    def __setitem__(self, key, value) -> None: ...  # type: ignore
    def __iter__(self) -> Iterator[AnyArray[_T]]: ...  # type: ignore
    @property
    def real(self) -> AnyArray[np.float32]: ...  # type: ignore
    @property
    def imag(self) -> AnyArray[np.float32]: ...  # type: ignore
    def conj(self) -> AnyArray[_T]: ...  # type: ignore
    @property
    def shape(self) -> tuple[int, ...]: ...  # type: ignore
    @property
    def ndim(self) -> int: ...
    @property
    def dtype(self) -> np.dtype[_T]: ...  # type: ignore
    def dot(self, other: AnyArray[_T]) -> AnyArray[_T]: ...  # type: ignore
    def astype(self, dtype: type[_T1]) -> AnyArray[_T1]: ...  # type: ignore
    @overload
    def mean(self, axis: None = None) -> _T: ...  # type: ignore
    @overload
    def mean(self, axis: int | tuple[int, ...]) -> AnyArray[_T]: ...  # type: ignore
    @overload
    def max(self, axis: None = None) -> _T: ...  # type: ignore
    @overload
    def max(self, axis: int | tuple[int, ...]) -> AnyArray[_T]: ...  # type: ignore
# fmt: on


class Backend:
    _default = "numpy"

    def __init__(self, name: str | None = None) -> None:
        if name is None:
            name = self._default
        if name == "numpy":
            from scipy import ndimage, fft  # type: ignore

            self._xp_ = np
            self._ndi_ = ndimage
            self._fft_ = fft
        elif name == "cupy":
            import cupy
            from cupyx.scipy import ndimage, fft

            self._xp_ = cupy
            self._ndi_ = ndimage
            self._fft_ = fft
        else:
            raise ValueError(f"Unknown backend {name}")

    @property
    def name(self) -> str:
        return self._xp_.__name__

    def __hash__(self) -> int:
        """Hash using the backend module."""
        return hash(self._xp_)

    def __repr__(self) -> str:
        return f"Backend<{self.name}>"

    def asnumpy(self, x: AnyArray[_T] | NDArray[_T]) -> NDArray[_T]:
        """Convert to numpy array."""
        if self._xp_ is np:
            return x  # type: ignore
        return x.get()  # type: ignore

    def maycopy(self, x: AnyArray[_T]) -> AnyArray[_T]:
        if self._xp_ is np:
            return x
        return x.copy()  # type: ignore

    @overload
    def array(self, x, dtype: type[_T] | np.dtype[_T]) -> AnyArray[_T]:
        ...

    @overload
    def array(self, x: AnyArray[_T] | NDArray[_T], dtype: None = None) -> AnyArray[_T]:
        ...

    def array(self, x, dtype=None):  # type: ignore
        """Convert to numpy array."""
        return self._xp_.array(x, dtype)  # type: ignore

    @overload
    def asarray(self, x, dtype: type[_T] | np.dtype[_T]) -> AnyArray[_T]:
        ...

    @overload
    def asarray(
        self, x: AnyArray[_T] | NDArray[_T], dtype: None = None
    ) -> AnyArray[_T]:
        ...

    def asarray(self, x, dtype=None):  # type: ignore
        """Convert to numpy array."""
        return self._xp_.asarray(x, dtype)  # type: ignore

    @overload
    def arange(self, *args, dtype: type[_T], **kwargs) -> AnyArray[_T]:
        ...

    @overload
    def arange(self, *args, dtype: None = None, **kwargs) -> AnyArray:
        ...

    def arange(self, *args, dtype=None, **kwargs):
        """Return evenly spaced values within a given interval."""
        return self._xp_.arange(*args, dtype=dtype, **kwargs)  # type: ignore

    @overload
    def linspace(
        self, start, stop, num: int, endpoint: bool, dtype: None = None
    ) -> AnyArray:
        ...

    @overload
    def linspace(
        self, start, stop, num: int, endpoint: bool, dtype: type[_T]
    ) -> AnyArray[_T]:
        ...

    def linspace(self, start, stop, num=50, endpoint=True, dtype=None):
        """Return evenly spaced numbers over a specified interval."""
        return self._xp_.linspace(start, stop, num, endpoint=endpoint, dtype=dtype)

    def zeros(
        self, shape: int | tuple[int, ...], dtype: type[_T] | np.dtype[_T] | None = None
    ) -> AnyArray[_T]:
        """Return a new array of given shape and type, filled with zeros."""
        return self._xp_.zeros(shape, dtype)  # type: ignore

    def full(
        self,
        shape: int | tuple[int, ...],
        fill_value: Any,
        dtype: type[_T] | np.dtype[_T] | None = None,
    ) -> AnyArray[_T]:
        """Return a new array of given shape and type, filled with fill_value."""
        return self._xp_.full(shape, fill_value, dtype=dtype)  # type: ignore

    @overload
    def sum(self, x: AnyArray[_T], axis: None = None) -> _T:
        ...

    @overload
    def sum(self, x: AnyArray[_T], axis: int | tuple[int, ...]) -> AnyArray[_T]:
        ...

    def sum(self, x, axis=None):
        """Return the sum of array elements over a given axis."""
        return self._xp_.sum(x, axis=axis)

    @overload
    def mean(self, x: AnyArray[_T], axis: None = None) -> _T:
        ...

    @overload
    def mean(self, x: AnyArray[_T], axis: int | tuple[int, ...]) -> AnyArray[_T]:
        ...

    def mean(self, x, axis=None):
        """Return the mean of array elements over a given axis."""
        return self._xp_.mean(x, axis=axis)

    def cumsum(self, x: AnyArray[_T], axis: int | None = None) -> AnyArray[_T]:
        return self._xp_.cumsum(x, axis=axis)  # type: ignore

    def sqrt(self, x: AnyArray[_T]) -> AnyArray[_T]:
        """Return the non-negative square-root of an array."""
        return self._xp_.sqrt(x)  # type: ignore

    def exp(self, x: AnyArray[_T]) -> AnyArray[_T]:
        """Return the exponential of an array."""
        return self._xp_.exp(x)  # type: ignore

    def pad(
        self,
        x: AnyArray[_T],
        pad_width: int | Sequence[int] | Sequence[tuple[int, int]],
        mode: str = "constant",
        **kwargs,
    ) -> AnyArray[_T]:
        """Pad an array."""
        return self._xp_.pad(x, pad_width, mode=mode, **kwargs)  # type: ignore

    def tensordot(
        self, a: AnyArray[_T], b: AnyArray[_T], axes: int | tuple[int, ...] = 2
    ) -> AnyArray[_T]:
        """Return tensor dot product of two arrays."""
        return self._xp_.tensordot(a, b, axes)  # type: ignore

    @overload
    def max(self, x: AnyArray[_T], axis: None = None) -> _T:
        ...

    @overload
    def max(self, x: AnyArray[_T], axis: int | tuple[int, ...]) -> AnyArray[_T]:
        ...

    def max(self, x, axis=None):
        """Return the maximum of an array or maximum along an axis."""
        return self._xp_.max(x, axis=axis)

    @overload
    def min(self, x: AnyArray[_T], axis: None = None) -> _T:
        ...

    @overload
    def min(self, x: AnyArray[_T], axis: int | tuple[int, ...]) -> AnyArray[_T]:
        ...

    def min(self, x, axis=None):
        """Return the minimum of an array or minimum along an axis."""
        return self._xp_.min(x, axis=axis)

    @overload
    def percentile(self, x: AnyArray[_T], q: float, axis: None = None) -> _T:
        ...

    @overload
    def percentile(
        self, x: AnyArray[_T], q: float, axis: int | tuple[int, ...]
    ) -> AnyArray[_T]:
        ...

    def percentile(self, x, q, axis=None):
        """Compute the q-th percentile of the data along the specified axis."""
        return self._xp_.percentile(x, q, axis=axis)

    @overload
    def argmin(self, x: AnyArray[_T], axis: None = None) -> np.intp:
        ...

    @overload
    def argmin(self, x: AnyArray[_T], axis: int | tuple[int, ...]) -> AnyArray[np.intp]:
        ...

    def argmin(self, x, axis=None):  # type: ignore
        return self._xp_.argmin(x, axis=axis)  # type: ignore

    @overload
    def argmax(self, x: AnyArray[_T], axis: None = None) -> np.intp:
        ...

    @overload
    def argmax(self, x: AnyArray[_T], axis: int | tuple[int, ...]) -> AnyArray[np.intp]:
        ...

    def argmax(self, x, axis=None):  # type: ignore
        return self._xp_.argmax(x, axis=axis)  # type: ignore

    def fix(self, x: AnyArray[_T]) -> AnyArray[_T]:
        """Round to nearest integer towards zero."""
        return self._xp_.fix(x)  # type: ignore

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
        x: AnyArray[np.complex64],
        s: tuple[int, int, int] | None = None,
        axes: int | tuple[int, ...] | None = None,
    ) -> AnyArray[np.float32]:
        """N-dimensional inverse FFT of real part."""
        return self._fft_.irfftn(x, s, axes)  # type: ignore

    def fftshift(self, x: AnyArray[_T], axes=None) -> AnyArray[_T]:
        """Shift zero-frequency component to center."""
        return self._xp_.fft.fftshift(x, axes=axes)  # type: ignore

    def ifftshift(self, x: AnyArray[_T], axes=None) -> AnyArray[_T]:
        """Inverse shift zero-frequency component to center."""
        return self._xp_.fft.ifftshift(x, axes=axes)  # type: ignore

    def fftfreq(self, n: int, d: float = 1.0) -> AnyArray[np.float_]:
        """Return the Discrete Fourier Transform sample frequencies."""
        return self._xp_.fft.fftfreq(n, d)  # type: ignore

    def meshgrid(
        self,
        *xi: AnyArray[_T],
        copy: bool = True,
        sparse: bool = False,
        indexing: Literal["xy", "ij"] = "xy",
    ) -> tuple[AnyArray[_T], ...]:
        """Return coordinate matrices from coordinate vectors."""
        return self._xp_.meshgrid(*xi, copy=copy, sparse=sparse, indexing=indexing)  # type: ignore

    @overload
    def indices(
        self, shape: tuple[int], dtype: type[_T] = np.int32
    ) -> tuple[AnyArray[_T]]:
        ...

    @overload
    def indices(
        self, shape: tuple[int, int], dtype: type[_T] = np.int32
    ) -> tuple[AnyArray[_T], AnyArray[_T]]:
        ...

    @overload
    def indices(
        self, shape: tuple[int, int, int], dtype: type[_T] = np.int32
    ) -> tuple[AnyArray[_T], AnyArray[_T], AnyArray[_T]]:
        ...

    @overload
    def indices(
        self, shape: tuple[int, ...], dtype: type[_T] = np.int32
    ) -> tuple[AnyArray[_T], ...]:
        ...

    def indices(self, shape, dtype=np.int32):  # type: ignore
        """Return an array representing the indices of a grid."""
        return self._xp_.indices(shape, dtype=dtype)  # type: ignore

    def unravel_index(self, indices, shape: tuple[int, ...]) -> AnyArray[np.intp]:
        """Converts a flat index into a tuple of coordinate arrays."""
        return self._xp_.asarray(self._xp_.unravel_index(indices, shape))  # type: ignore

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
            cval=float(cval),
            prefilter=prefilter,
        )  # type: ignore

    def spline_filter(
        self,
        input,
        order: int = 3,
        output: type[_T] = np.float64,
        mode: str = "mirror",
    ) -> AnyArray[_T]:
        return self._ndi_.spline_filter(
            self.asarray(input), order=order, output=output, mode=mode  # type: ignore
        )

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
            x, coords, order=order, mode=mode, cval=cval, prefilter=prefilter
        )  # type: ignore

    def sum_labels(
        self,
        arr: AnyArray[_T],
        labels: AnyArray[np.uint16],
        index: AnyArray[np.uint16],
    ) -> AnyArray[np.uint16]:
        return self._ndi_.sum_labels(arr, labels=labels, index=index)  # type: ignore

    def rotated_crop(
        self,
        subimg,
        mtx: NDArray[np.float32],
        shape: tuple[int, int, int],
        order: int,
        cval: float | Callable[[AnyArray[np.float32]], Any],
    ) -> AnyArray[np.float32]:
        if callable(cval):
            _cval = cval(subimg)
        else:
            _cval = cval

        out = self.affine_transform(
            subimg,
            matrix=self.asarray(mtx),
            output_shape=shape,
            order=order,
            prefilter=order > 1,
            mode="constant",
            cval=float(_cval),
        )
        return out

    def lowpass_filter_ft(self, img, cutoff, order: int = 2) -> AnyArray[np.complex64]:
        """Lowpass filter in Fourier space."""
        return _bandpass.lowpass_filter_ft(self, self.asarray(img), cutoff, order)

    def lowpass_filter(self, img, cutoff, order: int = 2) -> AnyArray[np.float32]:
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
def using_backend(name: str):
    """Context manager to temporarily change the default backend."""
    old_backend = Backend._default
    Backend._default = name
    try:
        yield
    finally:
        Backend._default = old_backend


def set_backend(name: str):
    """Set the default backend."""
    Backend._default = name
