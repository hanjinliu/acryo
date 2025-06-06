from __future__ import annotations

from typing import TypeVar
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

_T = TypeVar("_T", bound=np.number)


def axes_to_rotator(z: ArrayLike | None, y: ArrayLike) -> Rotation:
    """Determine the Rotation object that rotates [1, 0, 0] to z and [0, 1, 0] to y."""
    y0 = _normalize(np.atleast_2d(y))
    if z is None:
        z0 = _extract_orthogonal(y0, np.array([[1, 0, 0]]))
    else:
        z0 = _extract_orthogonal(y0, _normalize(np.atleast_2d(z)))
    x0 = _normalize(-np.cross(y0, z0))

    matrix = np.zeros((z0.shape[0], 3, 3))
    matrix[:, :, 0] = z0
    matrix[:, :, 1] = y0
    matrix[:, :, 2] = x0
    return Rotation.from_matrix(matrix)


def from_euler_xyz_coords(
    angles: ArrayLike, seq: str = "ZXZ", degrees: bool = False
) -> Rotation:
    """Create a rotator using zyx-coordinate system, from Euler angles."""
    seq = translate_euler(seq)
    angles = np.asarray(angles)
    return Rotation.from_euler(seq, angles[..., ::-1], degrees)


def translate_euler(seq: str) -> str:
    table = str.maketrans({"x": "z", "z": "x", "X": "Z", "Z": "X"})
    return seq[::-1].translate(table)


def _normalize(a: NDArray[_T]) -> NDArray[_T]:
    """Normalize vectors to length 1. Input must be (N, 3)."""
    return a / np.sqrt(np.sum(a**2, axis=1))[:, np.newaxis]


def _extract_orthogonal(a: NDArray[_T], b: NDArray[_T]) -> NDArray[_T]:
    """Extract component of b orthogonal to a."""
    a_norm = _normalize(a)
    return b - np.sum(a_norm * b, axis=1)[:, np.newaxis] * a_norm
