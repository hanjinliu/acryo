from __future__ import annotations

from typing import Callable, overload, TypeVar, Generic
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray
from acryo._types import nm
from acryo._utils import reshape_image

_R = TypeVar("_R", np.ndarray, "list[np.ndarray]")


class _Pipeline:
    def __init__(self, fn: Callable):
        self._func = fn
        self.__name__ = getattr(fn, "__name__", repr(fn))

    def with_name(self, name: str) -> Self:
        self.__name__ = name
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.__name__}>"

    def __radd__(self, other) -> Self:
        return self + other

    def __rsub__(self, other) -> Self:
        return self - other

    def __rmul__(self, other) -> Self:
        return self * other

    def __rtruediv__(self, other) -> Self:
        return self / other


class ImageProvider(_Pipeline, Generic[_R]):
    """Function that provides an image at a given scale."""

    def __init__(self, provider: Callable[[nm], _R]):
        super().__init__(provider)

    def provide(self, scale: nm) -> _R:
        """Provide an image at a given scale."""
        out = self._func(scale)
        assert_n_3d_arrays(out, self._func)
        return out

    def __call__(self, scale: nm) -> _R:
        return self.provide(scale)

    def __add__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) + other(scale)).with_name(
                f"({self.__name__} + {other.__name__})"
            )
        return self.__class__(lambda scale: self(scale) + other)

    def __sub__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) - other(scale)).with_name(
                f"({self.__name__} - {other.__name__})"
            )
        return self.__class__(lambda scale: self(scale) - other)

    def __mul__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) * other(scale)).with_name(
                f"{self.__name__} * {other.__name__}"
            )
        return self.__class__(lambda scale: self(scale) * other)

    def __truediv__(self, other) -> ImageProvider:
        if np.isscalar(other) and other == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) / other(scale)).with_name(
                f"{self.__name__} / {other.__name__}"
            )
        return self.__class__(lambda scale: self(scale) / other)

    def __eq__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) == other(scale)).with_name(
                f"{self.__name__} == {other.__name__}"
            )
        return self.__class__(lambda scale: self(scale) == other)

    def __ne__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) != other(scale)).with_name(
                f"{self.__name__} != {other.__name__}"
            )
        return self.__class__(lambda scale: self(scale) != other)

    def __lt__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(
                lambda scale: _lt(self(scale), other(scale))
            ).with_name(f"{self.__name__} < {other.__name__}")
        return self.__class__(lambda scale: self(scale) < other)

    def __le__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(
                lambda scale: _le(self(scale), other(scale))
            ).with_name(f"{self.__name__} <= {other.__name__}")
        return self.__class__(lambda scale: self(scale) <= other)

    def __gt__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(
                lambda scale: _gt(self(scale), other(scale))
            ).with_name(f"{self.__name__} > {other.__name__}")
        return self.__class__(lambda scale: self(scale) > other)

    def __ge__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(
                lambda scale: _ge(self(scale), other(scale))
            ).with_name(f"{self.__name__} >= {other.__name__}")
        return self.__class__(lambda scale: self(scale) >= other)

    def __neg__(self) -> ImageProvider:
        return self.__class__(lambda scale: -self(scale)).with_name(
            f"(-{self.__name__})"
        )


class ImageConverter(_Pipeline):
    """Functionthat convert an image."""

    def __init__(
        self, converter: Callable[[NDArray[np.float32], nm], NDArray[np.float32]]
    ):
        super().__init__(converter)

    def convert(self, image: NDArray[np.float32], scale: nm) -> NDArray[np.float32]:
        out = self._func(image, scale)
        _assert_3d_array(out, self._func)
        return out

    def __call__(self, image: NDArray[np.float32], scale: nm) -> NDArray[np.float32]:
        return self.convert(image, scale)

    @overload
    def compose(self, other: ImageProvider) -> ImageProvider: ...

    @overload
    def compose(self, other: ImageConverter) -> ImageConverter: ...

    def compose(self, other):
        """Function composition"""
        if isinstance(other, ImageProvider):
            fn = lambda scale: self(other(scale), scale)
        elif isinstance(other, ImageConverter):
            fn = lambda x, scale: self(other(x, scale), scale)
        else:
            raise TypeError("Cannot compose with a non-pipeline object.")
        return other.__class__(fn).with_name(f"{self.__name__}○{other.__name__}")

    __matmul__ = compose

    def with_scale(
        self, scale: nm, *, reshape_to: tuple[int, int, int] | None = None
    ) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]:
        """Partialize converter with a given scale."""

        def fn(img: NDArray[np.float32]) -> NDArray[np.float32]:
            arr = self(img, scale)
            if reshape_to is not None:
                arr = reshape_image(arr, reshape_to)
            return arr

        fn.__name__ = self._func.__name__
        return fn

    def __add__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(
                lambda x, scale: self(x, scale) + other(x, scale)
            ).with_name(f"({self.__name__} + {other.__name__})")
        elif isinstance(other, ImageProvider):
            return self.__class__(
                lambda x, scale: self(x, scale) + other(scale)
            ).with_name(f"({self.__name__} + {other.__name__})")
        return self.__class__(lambda x, scale: self(x, scale) + other)

    def __sub__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(
                lambda x, scale: self(x, scale) - other(x, scale)
            ).with_name(f"({self.__name__} - {other.__name__})")
        elif isinstance(other, ImageProvider):
            return self.__class__(
                lambda x, scale: self(x, scale) - other(scale)
            ).with_name(f"({self.__name__} - {other.__name__})")
        return self.__class__(lambda x, scale: self(x, scale) - other)

    def __mul__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(
                lambda x, scale: self(x, scale) * other(x, scale)
            ).with_name(f"({self.__name__} * {other.__name__})")
        elif isinstance(other, ImageProvider):
            return self.__class__(
                lambda x, scale: self(x, scale) * other(scale)
            ).with_name(f"({self.__name__} * {other.__name__})")
        return self.__class__(lambda x, scale: self(x, scale) * other)

    def __truediv__(self, other) -> ImageConverter:
        if np.isscalar(other) and other == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        if isinstance(other, ImageConverter):
            return self.__class__(
                lambda x, scale: self(x, scale) / other(x, scale)
            ).with_name(f"({self.__name__} / {other.__name__})")
        elif isinstance(other, ImageProvider):
            return self.__class__(
                lambda x, scale: self(x, scale) / other(scale)
            ).with_name(f"({self.__name__} / {other.__name__})")
        return self.__class__(lambda x, scale: self(x, scale) / other)

    def __eq__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(
                lambda x, scale: self(x, scale) == other(x, scale)
            ).with_name(f"({self.__name__} == {other.__name__})")
        elif isinstance(other, ImageProvider):
            return self.__class__(
                lambda x, scale: self(x, scale) == other(scale)
            ).with_name(f"({self.__name__} == {other.__name__})")
        return self.__class__(lambda x, scale: self(x, scale) == other)

    def __ne__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(
                lambda x, scale: self(x, scale) != other(x, scale)
            ).with_name(f"({self.__name__} != {other.__name__})")
        elif isinstance(other, ImageProvider):
            return self.__class__(
                lambda x, scale: self(x, scale) != other(scale)
            ).with_name(f"({self.__name__} != {other.__name__})")
        return self.__class__(lambda x, scale: self(x, scale) != other)

    def __gt__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(
                lambda x, scale: _gt(self(x, scale), other(x, scale))
            ).with_name(f"({self.__name__} > {other.__name__})")
        elif isinstance(other, ImageProvider):
            return self.__class__(
                lambda x, scale: _gt(self(x, scale), other(scale))
            ).with_name(f"({self.__name__} > {other.__name__})")
        return self.__class__(lambda x, scale: self(x, scale) > other)

    def __ge__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(
                lambda x, scale: _ge(self(x, scale), other(x, scale))
            ).with_name(f"({self.__name__} >= {other.__name__})")
        elif isinstance(other, ImageProvider):
            return self.__class__(
                lambda x, scale: _ge(self(x, scale), other(scale))
            ).with_name(f"({self.__name__} >= {other.__name__})")
        return self.__class__(lambda x, scale: self(x, scale) >= other)

    def __lt__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(
                lambda x, scale: _lt(self(x, scale), other(x, scale))
            ).with_name(f"({self.__name__} < {other.__name__})")
        elif isinstance(other, ImageProvider):
            return self.__class__(
                lambda x, scale: _lt(self(x, scale), other(scale))
            ).with_name(f"({self.__name__} < {other.__name__})")
        return self.__class__(lambda x, scale: self(x, scale) < other)

    def __le__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(
                lambda x, scale: _le(self(x, scale), other(x, scale))
            ).with_name(f"({self.__name__} <= {other.__name__})")
        elif isinstance(other, ImageProvider):
            return self.__class__(
                lambda x, scale: _le(self(x, scale), other(scale))
            ).with_name(f"({self.__name__} <= {other.__name__})")
        return self.__class__(lambda x, scale: self(x, scale) <= other)

    def __neg__(self) -> ImageConverter:
        return self.__class__(lambda x, scale: -self(x, scale))


def assert_n_3d_arrays(out, func):
    if isinstance(out, np.ndarray):
        return _assert_3d_array(out, func)
    for o in out:
        _assert_3d_array(o, func)


def _assert_3d_array(out, func):
    if not isinstance(out, np.ndarray):
        raise TypeError(
            f"Function {func!r} did not return a numpy.ndarray (got {type(out)})"
        )
    if out.ndim != 3:
        raise ValueError(f"Wrong image dimensionality: {out.shape}")


def _ge(a, b) -> NDArray[np.float32]:
    return np.greater_equal(a, b, dtype=np.float32)


def _le(a, b) -> NDArray[np.float32]:
    return np.less_equal(a, b, dtype=np.float32)


def _gt(a, b) -> NDArray[np.float32]:
    return np.greater(a, b, dtype=np.float32)


def _lt(a, b) -> NDArray[np.float32]:
    return np.less(a, b, dtype=np.float32)
