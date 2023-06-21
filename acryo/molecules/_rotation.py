from __future__ import annotations

from typing import TypeVar
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

_T = TypeVar("_T", bound=np.number)


def axes_to_rotator(z: ArrayLike, y: ArrayLike) -> Rotation:
    """Determine the Rotation object that rotates the z-axis to z and the y-axis to y."""
    y0 = _normalize(np.atleast_2d(y))
    rot_y = _get_align_rotator([[0, 1, 0]], y0)
    if z is None:
        z = np.array([[1, 0, 0]])
    else:
        z = np.atleast_2d(z)
    z0 = _normalize(np.atleast_2d(_extract_orthogonal(y0, z)))
    z0_trans = rot_y.apply(z0, inverse=True)
    rot_z = _get_align_rotator([[1, 0, 0]], z0_trans)
    return rot_y * rot_z


def _get_align_rotator(src, dst) -> Rotation:
    """R.apply(src) == dst. Both length must be 1."""
    cross = np.cross(src, dst)
    sin = norm = np.sqrt(np.sum(cross**2, axis=1, keepdims=True))
    cos = np.sum(src * dst, axis=1, keepdims=True)
    theta = np.arctan2(sin, cos)

    norm[norm == 0] = np.inf
    return Rotation.from_rotvec(cross / norm * theta)


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
