from __future__ import annotations
import numpy as np
from dask import array as da, delayed
import impy as ip
from typing import Callable


def multi_map_coordinates(
    input: ip.ImgArray | ip.LazyImgArray,
    coordinates: np.ndarray,
    order: int = 3,
    mode: str = "constant",
    cval: float | Callable[[ip.ImgArray], float] = 0.0,
) -> list[np.ndarray]:
    """
    Multiple map-coordinate in parallel.

    Result of this function is identical to following code.

    .. code-block:: python

        outputs = []
        for i in range(len(coordinates)):
            out = ndi.map_coordinates(input, coordinates[i], ...)
            outputs.append(out)

    """
    shape = input.shape
    coordinates = coordinates.copy()

    if coordinates.ndim != input.ndim + 2:
        if coordinates.ndim == input.ndim + 1:
            coordinates = coordinates[np.newaxis]
        else:
            raise ValueError(f"Coordinates have wrong dimension: {coordinates.shape}.")

    sl = []
    for i in range(coordinates.shape[1]):
        imin = int(np.min(coordinates[:, i])) - order
        imax = int(np.ceil(np.max(coordinates[:, i]))) + order + 1
        _sl, _pad = make_slice_and_pad(imin, imax, shape[i])
        sl.append(_sl)
        coordinates[:, i] -= _sl.start

    img = input[tuple(sl)]
    if isinstance(img, ip.LazyImgArray):
        img = img.compute()
    if callable(cval):
        cval = cval(img)
    input_img = img

    tasks = []
    for crds in coordinates:
        mapped = lazy_map_coordinates(
            input_img,
            coordinates=crds,
            order=order,
            mode=mode,
            cval=cval,
        )

        tasks.append(da.from_delayed(mapped, coordinates.shape[2:], dtype=np.float32))

    out = da.compute(tasks, scheduler=ip.Const["SCHEDULER"])[0]

    return np.stack(out, axis=0)


@delayed
def lazy_map_coordinates(
    input: ip.ImgArray,
    coordinates: np.ndarray,
    order: int = 3,
    mode: str = "constant",
    cval: float | Callable[[ip.ImgArray], float] = 0.0,
) -> np.ndarray:
    """Delayed version of ndi.map_coordinates."""
    return input.map_coordinates(
        coordinates=coordinates,
        order=order,
        mode=mode,
        cval=cval,
    )


def make_slice_and_pad(z0: int, z1: int, size: int) -> tuple[slice, tuple[int, int]]:
    """
    This function calculates what slicing and padding are needed when an array is sliced
    by ``z0:z1``. Array must be padded when z0 is negative or z1 is outside the array size.
    """
    z0_pad = z1_pad = 0
    if z0 < 0:
        z0_pad = -z0
        z0 = 0
    elif size < z0:
        raise ValueError(f"Specified size is {size} but need to slice at {z0}:{z1}.")

    if size < z1:
        z1_pad = z1 - size
        z1 = size
    elif z1 < 0:
        raise ValueError(f"Specified size is {size} but need to slice at {z0}:{z1}.")

    return slice(z0, z1), (z0_pad, z1_pad)
