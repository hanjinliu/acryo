from __future__ import annotations
from functools import lru_cache
import numpy as np
from acryo.backend import Backend, AnyArray


@lru_cache(maxsize=10)
def _build_mesh(
    shape: tuple[int, int, int],
    max_shifts: tuple[float, float, float],
    upsample: int,
    backend: Backend,
) -> AnyArray[np.float32]:
    upsampled_max_shifts = (np.asarray(max_shifts) * upsample).astype(np.int32)
    center = np.array(shape) / 2 - 0.5
    mesh = backend.meshgrid(
        *[
            backend.linspace(c - width / upsample, c + width / upsample, 2 * width + 1)
            for c, width in zip(center, upsampled_max_shifts)
        ],
        indexing="ij",
    )
    return backend.stack(mesh, axis=0)


def build_mesh(
    shape: tuple[int, ...],
    max_shifts: tuple[float, ...],
    upsample: int,
    backend: Backend,
) -> AnyArray[np.float32]:
    """Build a meshgrid for up-sampling.

    Parameters
    ----------
    shape : tuple[int, int, int]
        Shape of the original volume from which up-sampling will be applied.
    max_shifts : tuple[float, float, float]
        Maximum shifts in each direction in pixel.
    upsample : int
        Up-sampling factor.
    """
    return _build_mesh(shape, max_shifts, upsample, backend)
