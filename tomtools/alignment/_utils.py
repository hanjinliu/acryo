from __future__ import annotations
import itertools
import numpy as np
from ._types import Ranges, RangeLike
from ..molecules import from_euler


def _normalize_a_range(rng: RangeLike) -> RangeLike:
    if len(rng) != 2:
        raise TypeError("Range must be defined by (float, float).")
    max_rot, drot = rng
    return float(max_rot), float(drot)


def _normalize_ranges(rng: Ranges) -> Ranges:
    if isinstance(rng, tuple) and isinstance(rng[0], tuple):
        return tuple(_normalize_a_range(r) for r in rng)
    else:
        rng = _normalize_a_range(rng)
        return (rng,) * 3


def normalize_rotations(rotations: Ranges | None) -> np.ndarray:
    """
    Normalize various rotation expressions to quaternions.

    Parameters
    ----------
    rotations : tuple of float and float, or list of it, optional
        Rotation around each axis.

    Returns
    -------
    np.ndarray
        Corresponding quaternions in shape (N, 4).
    """
    if rotations is not None:
        rotations = _normalize_ranges(rotations)
        angles = []
        for max_rot, step in rotations:
            if step == 0:
                angles.append(np.zeros(1))
            else:
                n = int(max_rot / step)
                angles.append(np.linspace(-n * step, n * step, 2 * n + 1))

        quat: list[np.ndarray] = []
        for angs in itertools.product(*angles):
            quat.append(from_euler(np.array(angs), "zyx", degrees=True).as_quat())
        rotations = np.stack(quat, axis=0)
    else:
        rotations = np.array([[0.0, 0.0, 0.0, 1.0]])

    return rotations
