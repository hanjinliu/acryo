from __future__ import annotations
import itertools
from typing import Callable, Literal, Sequence, cast
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from acryo._types import Ranges, RangeLike, RotationType
from acryo.molecules import from_euler_xyz_coords
from acryo._typed_scipy import affine_transform


def _normalize_a_range(rng: RangeLike) -> RangeLike:
    if len(rng) != 2:
        raise TypeError(f"Range must be defined by (float, float), got {rng!r}")
    max_rot, drot = rng
    return float(max_rot), float(drot)


def _normalize_ranges(rng: RangeLike | Ranges) -> Ranges:
    if np.array(rng).ndim == 2:
        return tuple(_normalize_a_range(r) for r in rng)  # type: ignore
    else:
        rng_ = _normalize_a_range(rng)  # type: ignore
        return (rng_,) * 3


def _seq_of_max_and_step_to_quat(rotations: RangeLike | Ranges) -> NDArray[np.float32]:
    _rotations = _normalize_ranges(rotations)
    angles = []
    for max_rot, step in _rotations:
        if step == 0:
            angles.append(np.zeros(1))
        else:
            n = int(max_rot / step)
            angles.append(np.linspace(-n * step, n * step, 2 * n + 1))

    _quat: list[NDArray[np.float32]] = []
    for angs in itertools.product(*angles):
        _quat.append(
            from_euler_xyz_coords(np.array(angs), "zyx", degrees=True)
            .as_quat(canonical=False)
            .astype(np.float32, copy=False)
        )
    return np.stack(_quat, axis=0)


def normalize_rotations(rotations: RotationType | None) -> NDArray[np.floating]:
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
        quats = rotations.as_quat(canonical=False)
    elif rotations is not None:
        if not hasattr(rotations, "__iter__"):
            raise TypeError("rotations must be iterable")
        list_rot = list(rotations)
        if isinstance(list_rot[0], Rotation):
            list_rot = cast("list[Rotation]", list_rot)
            quats = np.stack(
                [
                    r.as_quat(canonical=False).astype(np.float32, copy=False)
                    for r in list_rot
                ],
                axis=0,
            )
        else:
            quats = _seq_of_max_and_step_to_quat(rotations)
    else:
        quats = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    return quats


def rotate(
    image: np.ndarray,
    degrees: tuple[float, float, float] | Sequence[float],
    order: int = 3,
    mode: Literal["constant", "nearest", "mirror", "wrap", "reflect"] = "constant",
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

    return affine_transform(
        image,
        matrix=matrix,
        order=order,
        mode=mode,
        cval=_cval,
        prefilter=order > 1,
    )


def euler_to_quat(degrees):
    return from_euler_xyz_coords(np.array(degrees), "zyx", degrees=True).as_quat(
        canonical=False
    )
