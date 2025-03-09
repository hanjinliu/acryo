from __future__ import annotations

from typing import Iterable, Any, Sequence
import numpy as np
from numpy.typing import NDArray
import polars as pl


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


def dict_iterrows(d: dict[str, Iterable[Any]]):
    """Generater similar to pl.DataFrame.iterrows().

    >>> dict_iterrows({'a': [1, 2, 3], 'b': [4, 5, 6]})

    will yield {'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}.
    """
    keys = d.keys()
    value_iters = [iter(v) for v in d.values()]

    dict_out = dict.fromkeys(keys, None)
    while True:
        try:
            for k, viter in zip(keys, value_iters):
                dict_out[k] = next(viter)
            yield dict_out
        except StopIteration:
            break


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
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    sl = rng.choice(np.arange(nmole), nmole // 2).tolist()
    indices0 = np.zeros(nmole, dtype=np.bool_)
    indices0[sl] = True
    indices1 = np.ones(nmole, dtype=np.bool_)
    indices1[sl] = False
    return indices0, indices1


def normalize_shape(a: int | Sequence[int], ndim: int):
    if isinstance(a, int):
        _output_shape = (a,) * ndim
    else:
        _output_shape = tuple(a)
    return _output_shape
