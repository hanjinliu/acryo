from __future__ import annotations

from typing import Callable, overload

import numpy as np
from numpy.typing import NDArray
from acryo._types import nm


class _Pipeline:
    def __init__(self, fn: Callable):
        self._func = fn

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._func.__name__})"


class ImageProvider(_Pipeline):
    """Function that provides an image at a given scale."""

    def __init__(self, provider: Callable[[nm], NDArray[np.float32]]):
        self._func = provider

    def __call__(self, scale: nm) -> NDArray[np.float32]:
        out = self._func(scale)
        _assert_3d_array(out, self._func)
        return out


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
    def __mul__(self, other: ImageProvider) -> ImageProvider:
        ...

    @overload
    def __mul__(self, other: ImageConverter) -> ImageConverter:
        ...

    def __mul__(self, other):
        """Function composition"""
        if isinstance(other, ImageProvider):
            fn = lambda scale: self(other(scale), scale)
        else:
            fn = lambda x, scale: self(other(x, scale), scale)
        return other.__class__(fn)

    def with_scale(
        self, scale: nm
    ) -> Callable[[NDArray[np.float32]], NDArray[np.float32]]:
        """Partialize converter with a given scale."""

        def fn(img: NDArray[np.float32]) -> NDArray[np.float32]:
            return self(img, scale)

        fn.__name__ = self._func.__name__
        return fn


def _assert_3d_array(out, func):
    if not isinstance(out, np.ndarray):
        raise TypeError(
            f"Function {func!r} did not return a numpy.ndarray (got {type(out)})"
        )
    if out.ndim != 3:
        raise ValueError(f"Wrong image dimensionality: {out.shape}")
