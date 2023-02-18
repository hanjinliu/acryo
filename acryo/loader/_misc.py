from __future__ import annotations

from typing import Iterable, Any
import numpy as np
import polars as pl


def get_feature_list(corr_max, local_shifts, rotvec) -> list[pl.Series]:
    features = [
        pl.Series("score", corr_max),
        pl.Series("shift-z", np.round(local_shifts[:, 0], 2)),
        pl.Series("shift-y", np.round(local_shifts[:, 1], 2)),
        pl.Series("shift-x", np.round(local_shifts[:, 2], 2)),
        pl.Series("rotvec-z", np.round(rotvec[:, 0], 5)),
        pl.Series("rotvec-y", np.round(rotvec[:, 1], 5)),
        pl.Series("rotvec-x", np.round(rotvec[:, 2], 5)),
    ]
    return features


def dict_iterrows(d: dict[str, Iterable[Any]]):
    """
    Generater similar to pl.DataFrame.iterrows().

    >>> _dict_iterrows({'a': [1, 2, 3], 'b': [4, 5, 6]})

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


def allocate(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # shift in local Cartesian
    local_shifts = np.zeros((size, 3))

    # maximum ZNCC
    corr_max = np.zeros(size)

    # rotation (quaternion) in local Cartesian
    local_rot = np.zeros((size, 4))
    local_rot[:, 3] = 1  # identity map in quaternion

    return local_shifts, local_rot, corr_max
