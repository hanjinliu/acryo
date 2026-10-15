from __future__ import annotations

from typing import Sequence, SupportsInt, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import polars as pl
from dask import array as da
from scipy.spatial.transform import Rotation
from acryo._types import nm

if TYPE_CHECKING:
    from acryo.backend import Backend, AnyArray
    from acryo.tilt import TiltSeriesModel


def get_feature_list(corr_max, local_shifts, rotvec) -> list[pl.Series]:
    features = [
        pl.Series("score", corr_max),
        pl.Series("align-dz", np.round(local_shifts[:, 0], 2)),
        pl.Series("align-dy", np.round(local_shifts[:, 1], 2)),
        pl.Series("align-dx", np.round(local_shifts[:, 2], 2)),
        pl.Series("align-dzrot", np.round(rotvec[:, 0], 5)),
        pl.Series("align-dyrot", np.round(rotvec[:, 1], 5)),
        pl.Series("align-dxrot", np.round(rotvec[:, 2], 5)),
    ]
    return features


def allocate(
    size: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    # shift in local Cartesian
    local_shifts = np.zeros((size, 3), dtype=np.float32)

    # maximum ZNCC
    corr_max = np.zeros(size, dtype=np.float32)

    # rotation (quaternion) in local Cartesian
    local_rot = np.zeros((size, 4), dtype=np.float32)
    local_rot[:, 3] = 1  # identity map in quaternion

    return local_shifts, local_rot, corr_max


def random_splitter(
    rng: np.random.Generator,
    nmole: int,
    nsplit: int = 2,
) -> list[NDArray[np.bool_]]:
    indices = np.arange(nmole)
    rng.shuffle(indices)
    outs = []
    for i in range(nsplit):
        mask = indices % nsplit == i
        outs.append(mask)
    return outs


def normalize_shape(a: int | Sequence[int], ndim: int):
    if isinstance(a, SupportsInt):
        _output_shape = (int(a),) * ndim
    else:
        _output_shape = tuple(int(each) for each in a)
    return _output_shape


def normalize_max_shifts(x: nm | tuple[nm, nm, nm]) -> tuple[nm, nm, nm]:
    if hasattr(x, "__iter__"):
        tup = tuple(float(x0) for x0 in x)  # type: ignore
        if len(tup) != 3:
            raise ValueError(
                "max_shifts must be a 3-tuple if multiple values are given."
            )
        return tup
    return (float(x),) * 3  # type: ignore


def wedge_weighted_ft(
    img: NDArray[np.float32],
    quat: NDArray[np.float64],
    tilt_model: TiltSeriesModel,
    backend: Backend,
) -> AnyArray[np.complex64]:
    """Return the masked spectrum of a subvolume stacked with the mask itself.

    The spectrum is in the layout of ``rfftn``. The missing wedge mask is symmetric
    about the origin, so the last axis can be cropped in the same way.
    """
    xp = backend
    shape = img.shape
    mask = tilt_model.create_mask(Rotation.from_quat(quat), shape)
    mask = xp.asarray(mask[..., : shape[2] // 2 + 1], dtype=np.float32)
    ft = xp.rfftn(xp.asarray(img)) * mask
    return xp.stack([ft, mask.astype(np.complex64)], axis=0)


def lazy_average(
    stack: da.Array,
    output_shape: tuple[int, int, int],
    backend: Backend,
    wedge_weighting: bool = False,
    wiener_constant: float = 0.1,
) -> da.Array:
    """Construct a lazy average of the stack of subvolumes.

    If ``wedge_weighting`` is true, ``stack`` must be the stack of the outputs of
    `wedge_weighted_ft`.
    """
    if not wedge_weighting:
        return stack.mean(axis=0)
    if wiener_constant < 0:
        raise ValueError(
            f"wiener_constant must be non-negative, got {wiener_constant}."
        )
    return da.map_blocks(
        _wiener_filter,
        stack.sum(axis=0),
        regularization=wiener_constant * stack.shape[0],
        output_shape=output_shape,
        backend=backend,
        drop_axis=0,
        chunks=tuple((s,) for s in output_shape),
        meta=np.empty((0, 0, 0), dtype=np.float32),
    )


def _wiener_filter(
    summed: AnyArray[np.complex64],
    regularization: float,
    output_shape: tuple[int, int, int],
    backend: Backend,
) -> AnyArray[np.float32]:
    xp = backend
    ft_sum, mask_sum = summed[0], summed[1].real
    denom = mask_sum + regularization
    # never sampled by any subvolume. The numerator is also zero.
    denom[denom == 0] = 1.0
    return xp.irfftn(ft_sum / denom, s=output_shape).astype(np.float32)
