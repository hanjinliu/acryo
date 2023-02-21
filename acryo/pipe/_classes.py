from __future__ import annotations

from typing import Callable, overload
from typing_extensions import Self

import numpy as np
from numpy.typing import NDArray
from acryo._types import nm


class _Pipeline:
    def __init__(self, fn: Callable):
        self._func = fn

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._func.__name__})"

    def __radd__(self, other) -> Self:
        return self + other

    def __rsub__(self, other) -> Self:
        return self - other

    def __rmul__(self, other) -> Self:
        return self * other

    def __rtruediv__(self, other) -> Self:
        return self / other


class ImageProvider(_Pipeline):
    """Function that provides an image at a given scale."""

    def __init__(self, provider: Callable[[nm], NDArray[np.float32]]):
        self._func = provider

    def __call__(self, scale: nm) -> NDArray[np.float32]:
        out = self._func(scale)
        _assert_3d_array(out, self._func)
        return out

    def __add__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) + other(scale))
        return self.__class__(lambda scale: self(scale) + other)

    def __sub__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) - other(scale))
        return self.__class__(lambda scale: self(scale) - other)

    def __mul__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) * other(scale))
        return self.__class__(lambda scale: self(scale) * other)

    def __truediv__(self, other) -> ImageProvider:
        if np.isscalar(other) and other == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) / other(scale))
        return self.__class__(lambda scale: self(scale) / other)

    def __eq__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) == other(scale))
        return self.__class__(lambda scale: self(scale) == other)

    def __ne__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) != other(scale))
        return self.__class__(lambda scale: self(scale) != other)

    def __lt__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) < other(scale))
        return self.__class__(lambda scale: self(scale) < other)

    def __le__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) <= other(scale))
        return self.__class__(lambda scale: self(scale) <= other)

    def __gt__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) > other(scale))
        return self.__class__(lambda scale: self(scale) > other)

    def __ge__(self, other) -> ImageProvider:
        if isinstance(other, ImageProvider):
            return self.__class__(lambda scale: self(scale) >= other(scale))
        return self.__class__(lambda scale: self(scale) >= other)

    def __neg__(self) -> ImageProvider:
        return self.__class__(lambda scale: -self(scale))


class ImageConverter(_Pipeline):
    """Functionthat convert an image."""

    def __init__(
        self, converter: Callable[[NDArray[np.float32], nm], NDArray[np.float32]]
    ):
        self._func = converter

    def __call__(self, image: NDArray[np.float32], scale: nm) -> NDArray[np.float32]:
        out = self._func(image, scale)
        _assert_3d_array(out, self._func)
        return out

    @overload
    def compose(self, other: ImageProvider) -> ImageProvider:
        ...

    @overload
    def compose(self, other: ImageConverter) -> ImageConverter:
        ...

    def compose(self, other):
        """Function composition"""
        if isinstance(other, ImageProvider):
            fn = lambda scale: self(other(scale), scale)
        elif isinstance(other, ImageConverter):
            fn = lambda x, scale: self(other(x, scale), scale)
        else:
            raise TypeError("Cannot compose with a non-pipeline object.")
        return other.__class__(fn)

    __matmul__ = compose

    def with_scale(
        self, scale: nm
    ) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]:
        """Partialize converter with a given scale."""

        def fn(img: NDArray[np.float32]) -> NDArray[np.float32]:
            return self(img, scale)

        fn.__name__ = self._func.__name__
        return fn

    def __add__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(lambda x, scale: self(x, scale) + other(x, scale))
        elif isinstance(other, ImageProvider):
            return self.__class__(lambda x, scale: self(x, scale) + other(scale))
        return self.__class__(lambda x, scale: self(x, scale) + other)

    def __sub__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(lambda x, scale: self(x, scale) - other(x, scale))
        elif isinstance(other, ImageProvider):
            return self.__class__(lambda x, scale: self(x, scale) - other(scale))
        return self.__class__(lambda x, scale: self(x, scale) - other)

    def __mul__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(lambda x, scale: self(x, scale) * other(x, scale))
        elif isinstance(other, ImageProvider):
            return self.__class__(lambda x, scale: self(x, scale) * other(scale))
        return self.__class__(lambda x, scale: self(x, scale) * other)

    def __truediv__(self, other) -> ImageConverter:
        if np.isscalar(other) and other == 0:
            raise ZeroDivisionError("Cannot divide by zero.")
        if isinstance(other, ImageConverter):
            return self.__class__(lambda x, scale: self(x, scale) / other(x, scale))
        elif isinstance(other, ImageProvider):
            return self.__class__(lambda x, scale: self(x, scale) / other(scale))
        return self.__class__(lambda x, scale: self(x, scale) / other)

    def __eq__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(lambda x, scale: self(x, scale) == other(x, scale))
        elif isinstance(other, ImageProvider):
            return self.__class__(lambda x, scale: self(x, scale) == other(scale))
        return self.__class__(lambda x, scale: self(x, scale) == other)

    def __ne__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(lambda x, scale: self(x, scale) != other(x, scale))
        elif isinstance(other, ImageProvider):
            return self.__class__(lambda x, scale: self(x, scale) != other(scale))
        return self.__class__(lambda x, scale: self(x, scale) != other)

    def __gt__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(lambda x, scale: self(x, scale) > other(x, scale))
        elif isinstance(other, ImageProvider):
            return self.__class__(lambda x, scale: self(x, scale) > other(scale))
        return self.__class__(lambda x, scale: self(x, scale) > other)

    def __ge__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(lambda x, scale: self(x, scale) >= other(x, scale))
        elif isinstance(other, ImageProvider):
            return self.__class__(lambda x, scale: self(x, scale) >= other(scale))
        return self.__class__(lambda x, scale: self(x, scale) >= other)

    def __lt__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(lambda x, scale: self(x, scale) < other(x, scale))
        elif isinstance(other, ImageProvider):
            return self.__class__(lambda x, scale: self(x, scale) < other(scale))
        return self.__class__(lambda x, scale: self(x, scale) < other)

    def __le__(self, other) -> ImageConverter:
        if isinstance(other, ImageConverter):
            return self.__class__(lambda x, scale: self(x, scale) <= other(x, scale))
        elif isinstance(other, ImageProvider):
            return self.__class__(lambda x, scale: self(x, scale) <= other(scale))
        return self.__class__(lambda x, scale: self(x, scale) <= other)

    def __neg__(self) -> ImageConverter:
        return self.__class__(lambda x, scale: -self(x, scale))


def _assert_3d_array(out, func):
    if not isinstance(out, np.ndarray):
        raise TypeError(
            f"Function {func!r} did not return a numpy.ndarray (got {type(out)})"
        )
    if out.ndim != 3:
        raise ValueError(f"Wrong image dimensionality: {out.shape}")
