from __future__ import annotations

from functools import lru_cache
import math
import numpy as np
from numpy.typing import NDArray


@lru_cache(maxsize=8)
def get_norms_y(
    tilt_range: tuple[float, float]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    degmin, degmax = tilt_range
    ang0 = math.pi - math.radians(degmin)
    ang1 = math.pi - math.radians(degmax)
    return (
        np.array([math.cos(ang0), 0.0, math.sin(ang0)], dtype=np.float32),
        np.array([math.cos(ang1), 0.0, math.sin(ang1)], dtype=np.float32),
    )


@lru_cache(maxsize=8)
def get_norms_x(
    tilt_range: tuple[float, float]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    degmin, degmax = tilt_range
    ang0 = math.pi - math.radians(degmin)
    ang1 = math.pi - math.radians(degmax)
    return (
        np.array([math.cos(ang0), math.sin(ang0), 0.0], dtype=np.float32),
        np.array([math.cos(ang1), math.sin(ang1), 0.0], dtype=np.float32),
    )


@lru_cache(maxsize=32)
def get_indices(shape: tuple[int, int, int]) -> NDArray[np.float32]:
    inds = np.indices(shape, dtype=np.float32)
    for ind, s in zip(inds, shape):
        # Zero frequency is at s // 2 after fftshift, so the centered indices are
        # shifted back by ifftshift (not fftshift, which differs for odd sizes).
        ind -= s // 2
    return np.fft.ifftshift(np.stack(list(inds), axis=-1), axes=(0, 1, 2))
