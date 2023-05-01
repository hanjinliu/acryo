from __future__ import annotations

from typing import TypeVar
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

_T = TypeVar("_T", bound=np.number)


def axes_to_rotator(z, y) -> Rotation:
    ref = _normalize(np.atleast_2d(y))

    n = ref.shape[0]
    yx = np.arctan2(ref[:, 2], ref[:, 1])
    zy = np.arctan(-ref[:, 0] / np.abs(ref[:, 1]))

    rot_vec_yx = np.zeros((n, 3))
    rot_vec_yx[:, 0] = yx
    rot_yx = Rotation.from_rotvec(rot_vec_yx)

    rot_vec_zy = np.zeros((n, 3))
    rot_vec_zy[:, 2] = zy
    rot_zy = Rotation.from_rotvec(rot_vec_zy)

    rot1 = rot_yx * rot_zy

    if z is None:
        return rot1

    vec = _normalize(np.atleast_2d(_extract_orthogonal(ref, z)))

    vec_trans = rot1.apply(vec, inverse=True)  # in zx-plane

    thetas = np.arctan2(vec_trans[..., 0], vec_trans[..., 2]) - np.pi / 2

    rot_vec_zx = np.zeros((n, 3))
    rot_vec_zx[:, 1] = thetas
    rot2 = Rotation.from_rotvec(rot_vec_zx)

    return rot1 * rot2


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
