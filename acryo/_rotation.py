from __future__ import annotations
import itertools
from typing import Callable, Sequence
import numpy as np
from scipy.spatial.transform import Rotation
from scipy import ndimage as ndi

from acryo._types import Ranges, RangeLike
from acryo.molecules import from_euler_xyz_coords


def _normalize_a_range(rng: RangeLike) -> RangeLike:
    if len(rng) != 2:
        raise TypeError(f"Range must be defined by (float, float), got {rng!r}")
    max_rot, drot = rng
    return float(max_rot), float(drot)


def _normalize_ranges(rng: RangeLike | Ranges) -> Ranges:
    if isinstance(rng, (tuple, list)) and isinstance(rng[0], tuple):
        return tuple(_normalize_a_range(r) for r in rng)  # type: ignore
    else:
        rng_ = _normalize_a_range(rng)  # type: ignore
        return (rng_,) * 3


def normalize_rotations(rotations: Ranges | Rotation | None) -> np.ndarray:
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
    if isinstance(rotations, Rotation):
        quat = rotations.as_quat()
    elif rotations is not None:
        _rotations = _normalize_ranges(rotations)
        angles = []
        for max_rot, step in _rotations:
            if step == 0:
                angles.append(np.zeros(1))
            else:
                n = int(max_rot / step)
                angles.append(np.linspace(-n * step, n * step, 2 * n + 1))

        quat: list[np.ndarray] = []
        for angs in itertools.product(*angles):
            quat.append(
                from_euler_xyz_coords(np.array(angs), "zyx", degrees=True).as_quat()
            )
        quats = np.stack(quat, axis=0)
    else:
        quats = np.array([[0.0, 0.0, 0.0, 1.0]])

    return quats


def rotate(
    image: np.ndarray,
    degrees: tuple[float, float, float] | Sequence[float],
    order: int = 3,
    mode="constant",
    cval: Callable | float = np.mean,
):
    from acryo._utils import compose_matrices

    quat = euler_to_quat(degrees)
    rotator = Rotation.from_quat(quat).inv()
    matrix = compose_matrices(
        np.array(image.shape[-image.ndim :]) / 2 - 0.5, [rotator]
    )[0]
    if callable(cval):
        _cval = cval(image)
    else:
        _cval = cval

    return ndi.affine_transform(
        image,
        matrix=matrix,
        order=order,
        mode=mode,
        cval=_cval,
        prefilter=order > 1,
    )


def euler_to_quat(degrees):
    return from_euler_xyz_coords(np.array(degrees), "zyx", degrees=True).as_quat()
