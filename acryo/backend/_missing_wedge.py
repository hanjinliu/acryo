from __future__ import annotations

from functools import lru_cache
import math
from typing import TYPE_CHECKING
from scipy.spatial.transform import Rotation
from acryo._types import degree
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ._api import AnyArray, Backend


def missing_wedge_mask(
    backend: Backend,
    rotator: Rotation,
    tilt_range: tuple[degree, degree],
    shape: tuple[int, int, int],
) -> AnyArray[np.float32]:
    """
    Create a binary mask that covers tomographical missing wedge.

    Note that the mask is not shifted to the center of the Fourier domain.
    ``np.fft.fftn(img) * mask`` will be the correct way to apply the mask.

    Parameters
    ----------
    rotator : Rotation
        The rotation object that describes the direction of the mask.
    tilt_range : tuple of float
        The range of tilt angles in degrees.
    shape : tuple of int
        The shape of the mask.

    Returns
    -------
    np.ndarray or float
        Missing wedge mask. If ``tilt_range`` is None, 1 will be returned.
    """
    normal0, normal1 = _get_unrotated_normals(tilt_range)
    shape_vector = np.array(shape, dtype=np.float32)
    rotator_inv = rotator.inv()
    normal0 = rotator_inv.apply(normal0 * shape_vector)
    normal1 = rotator_inv.apply(normal1 * shape_vector)
    vectors = _get_indices(shape, backend)
    dot0 = vectors.dot(backend.asarray(normal0))
    dot1 = vectors.dot(backend.asarray(normal1))
    missing = dot0 * dot1 <= 0
    return missing


@lru_cache(maxsize=8)
def _get_unrotated_normals(
    tilt_range: tuple[degree, degree]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    radmin, radmax = np.deg2rad(tilt_range)
    ang0 = np.pi - radmin
    ang1 = np.pi - radmax
    return (
        np.array([np.cos(ang0), 0, np.sin(ang0)], dtype=np.float32),
        np.array([np.cos(ang1), 0, np.sin(ang1)], dtype=np.float32),
    )


@lru_cache(maxsize=32)
def _get_indices(shape: tuple[int, int, int], backend: Backend) -> AnyArray[np.float32]:
    inds = backend.indices(shape, dtype=np.float32)
    for ind, s in zip(inds, shape):
        # Note that the shifts in indices must resemble the shifts in fftshift.
        ind -= math.ceil(s / 2)
    return backend.fftshift(
        backend.stack(list(inds), axis=-1), axes=(0, 1, 2)
    )  # type: ignore
