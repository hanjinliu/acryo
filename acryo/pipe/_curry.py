from __future__ import annotations

from typing import Callable, TypeVar
from typing_extensions import ParamSpec, Concatenate
from acryo.loader import ImageProvider, ImageConverter
import inspect

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
        return ImageProvider(lambda scale: fn(scale, *args, **kwargs))

    _update_wrapper(inner, fn)
    return inner


def converter_function(
    fn: Callable[Concatenate[float, _P], _R]
) -> Callable[_P, ImageConverter]:
    """
    Convert a function into a curried function that returns a image converter.

    Examples
    --------
    >>> from scipy import ndimage as ndi
    >>> @converter_function
    ... def gaussian_filter(img, sigma):
    ...     return ndi.gaussian_filter(img, sigma)
    >>> converter = gaussian_filter(1.5)
    >>> converter(arr)  # return a Gaussian filtered `arr`

    """

    def inner(*args, **kwargs):
        return ImageConverter(lambda img: fn(img, *args, **kwargs))

    _update_wrapper(inner, fn)
    return inner


def _update_wrapper(f, wrapped):
    for name in ("__module__", "__name__", "__qualname__", "__doc__"):
        try:
            value = getattr(wrapped, name)
        except AttributeError:
            pass
        else:
            setattr(f, name, value)
    annot = getattr(wrapped, "__annotations__", {}).copy()
    if len(annot) == 0:
        return f
    try:
        args = inspect.getargs(wrapped.__code__)
    except Exception:
        pass
    else:
        first = args.args[0]
        annot.pop(first)
    return f
