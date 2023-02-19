from __future__ import annotations

from typing import Any, Callable
import inspect

import numpy as np
from numpy.typing import NDArray
from acryo._types import nm


class ImageProvider:
    """Function that provides an image at a given scale."""

    def __init__(self, provider: Callable[[nm], NDArray[np.float32]]):
        self._provider = _assert_0_or_1_args(provider)

    def __call__(self, scale: nm) -> NDArray[np.float32]:
        out = self._provider(scale)
        _assert_3d_array(out, self._provider)
        return out


class ImageConverter:
    """Functionthat convert an image."""

    def __init__(self, converter: Callable[[NDArray[np.float32]], NDArray[np.float32]]):
        self._converter = _assert_0_or_1_args(converter)

    def __call__(self, image: NDArray[np.float32]) -> NDArray[np.float32]:
        out = self._converter(image)
        _assert_3d_array(out, self._converter)
        return out

    def __mul__(self, other: ImageProvider | ImageConverter):
        """Function composition"""
        fn = lambda x: self(other(x))
        return other.__class__(fn)


def _assert_3d_array(out, func):
    if not isinstance(out, np.ndarray):
        raise TypeError(
            f"Function {func!r} did not return a numpy.ndarray (got {type(out)})"
        )
    if out.ndim != 3:
        raise ValueError(f"Wrong image dimensionality: {out.shape}")


def _assert_0_or_1_args(func: Callable) -> Callable[[Any], Any]:
    nargs = sum(
        1
        for p in inspect.signature(func).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    if nargs == 0:
        return lambda x: func()
    elif nargs == 1:
        return func
    else:
        raise TypeError(
            "Expected zero or one positional argument but input function require at "
            "least two positional arguments"
        )
