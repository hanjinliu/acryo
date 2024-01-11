from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

from acryo.backend import Backend, AnyArray

UPSAMPLE = 20


def upsample(
    res: AnyArray,
    res_ori: AnyArray,
    max_shifts: tuple[float, float, float],
    pad_width_eff: tuple[float, float, float],
    backend: Backend,
) -> tuple[NDArray[np.float32], float]:
    maxima = backend.asnumpy(backend.unravel_index(backend.argmax(res), res.shape))
    midpoints = np.asarray(res.shape, dtype=np.int32) // 2
    coords, local_offset = _create_mesh(
        maxima,
        max_shifts,
        midpoints.astype(np.float32),
        pad_width_eff,
        backend,
    )
    local_response = backend.map_coordinates(
        res_ori, coords, order=3, mode="constant", cval=-1.0, prefilter=True
    )
    local_maxima = backend.asnumpy(
        backend.unravel_index(backend.argmax(local_response), local_response.shape)
    )
    corr = backend.asnumpy(local_response[tuple(local_maxima)])
    loc_shift = local_maxima / UPSAMPLE + local_offset
    shifts = maxima - midpoints + loc_shift
    return shifts, corr  # type: ignore


def _create_mesh(
    maxima: NDArray[np.intp],
    max_shifts: tuple[float, float, float],
    midpoints: NDArray[np.float32],
    pad_width_eff: tuple[float, float, float],
    backend: Backend,
):
    """
    Create a 3 pixel x 3 pixel (if not upsampled) mesh for image upsampling.
    """
    shifts = np.asarray(maxima, dtype=np.float32) - midpoints
    _max_shifts = np.asarray(max_shifts, dtype=np.float32)
    left = -shifts - _max_shifts
    right = -shifts + _max_shifts
    local_shifts = [
        [
            int(round(max(float(shiftl), -1.0) * UPSAMPLE)),
            int(round(min(float(shiftr), 1.0) * UPSAMPLE)),
        ]
        for shiftl, shiftr in zip(left, right)
    ]
    mesh = backend.stack(
        backend.meshgrid(
            *[
                backend.arange(s0, s1 + 1) / UPSAMPLE + m + w
                for (s0, s1), m, w in zip(local_shifts, maxima, pad_width_eff)
            ],  # type: ignore
            indexing="ij",
        ),
        axis=0,
    )
    offset = backend.array([s0 for s0, s1 in local_shifts], dtype=np.float32) / UPSAMPLE
    return mesh, offset
