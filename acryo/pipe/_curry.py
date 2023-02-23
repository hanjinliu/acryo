from __future__ import annotations

from typing import Callable, TypeVar, Any
from typing_extensions import ParamSpec, Concatenate
import inspect
import numpy as np
from acryo.pipe._classes import ImageProvider, ImageConverter

_P = ParamSpec("_P")
_R = TypeVar("_R")


def provider_function(
    fn: Callable[Concatenate[float, _P], _R]
) -> Callable[_P, ImageProvider]:
    """
    Convert a function into a curried function that returns a image provider.

    Examples
    --------
    >>> @provider_function
    ... def provide_random_image(scale, shape):
    ...     return np.random.random(shape)
    >>> provider = provide_random_image((10, 20, 30))
    >>> provider(0.18)  # return a (10, 20, 30) array

    """

    def inner(*args, **kwargs):
        _fn = _assert_1_arg(fn)
        return ImageProvider(lambda scale: _fn(scale, *args, **kwargs)).with_name(
            _format_args(fn, *args, **kwargs)
        )

    _update_wrapper(inner, fn, npop=1)
    return inner


def converter_function(
    fn: Callable[Concatenate[np.ndarray, float, _P], _R]
) -> Callable[_P, ImageConverter]:
    """
    Convert a function into a curried function that returns a image converter.

    Input function must accept `fn(img, scale)`.

    Examples
    --------
    >>> from scipy import ndimage as ndi
    >>> @converter_function
    ... def gaussian_filter(img, scale, sigma):
    ...     return ndi.gaussian_filter(img, sigma / scale)
    >>> converter = gaussian_filter(1.5)
    >>> converter(arr)  # return a Gaussian filtered `arr`

    """

    def inner(*args, **kwargs):
        _fn = _assert_2_args(fn)
        return ImageConverter(
            lambda img, scale: _fn(img, scale, *args, **kwargs)
        ).with_name(_format_args(fn, *args, **kwargs))

    _update_wrapper(inner, fn, npop=2)
    return inner


def _update_wrapper(f, wrapped, npop: int):
    _update_attr(f, wrapped)
    annot = getattr(wrapped, "__annotations__", {}).copy()
    if len(annot) == 0:
        return f
    try:
        args = inspect.getargs(wrapped.__code__)
    except Exception:
        pass
    else:
        for i in range(npop):
            name = args.args[i]
            annot.pop(name, None)
    return f


def _assert_1_arg(func: Callable) -> Callable[[Any], Any]:
    nargs = sum(
        1
        for p in inspect.signature(func).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    if nargs == 0:
        out = lambda x: func()
        _update_attr(out, func)
    else:
        return func


def _assert_2_args(func: Callable) -> Callable[[Any, Any], Any]:
    nargs = sum(
        1
        for p in inspect.signature(func).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    if nargs == 0:
        out = lambda x0, x1: func()
    elif nargs == 1:
        out = lambda x0, x1: func(x0)
    else:
        return func
    _update_attr(out, func)
    return out


def _update_attr(f, wrapped):
    for name in ("__module__", "__name__", "__qualname__", "__doc__"):
        try:
            value = getattr(wrapped, name)
        except AttributeError:
            pass
        else:
            setattr(f, name, value)


# formatter
def _format(arg: Any) -> str:
    if isinstance(arg, (float, complex)):
        return format(arg, ".2f")
    elif hasattr(arg, "__array__"):
        return "..."
    else:
        return repr(arg)


def _format_args(fn: Callable, *args, **kwargs):
    _args = [_format(arg) for arg in args]
    _kwargs = [f"{k}={_format(v)}" for k, v in kwargs.items()]

    s = ", ".join(_args + _kwargs)
    return f"{fn.__name__}({s})"
