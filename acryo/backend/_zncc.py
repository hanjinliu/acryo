from __future__ import annotations
from functools import lru_cache
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len

from acryo._types import pixel
from acryo.backend import Backend, AnyArray

# Normalized cross correlation


def ncc_landscape(
    img0: AnyArray[np.float32],
    img1: AnyArray[np.float32],
    max_shifts: tuple[float, ...],
    backend: Backend,
) -> AnyArray[np.float32]:
    if max_shifts is not None:
        max_shifts = tuple(max_shifts)
    pad_width = _get_padding_width(max_shifts)
    padimg = np.pad(img0, pad_width=pad_width, mode="constant", constant_values=0)
    return ncc_landscape_no_pad(padimg, img1, backend=backend)


def ncc_landscape_no_pad(
    img0: AnyArray[np.float32],
    img1: AnyArray[np.float32],
    backend: Backend,
) -> AnyArray[np.float32]:
    corr = fftconvolve(img0, img1[::-1, ::-1, ::-1], backend)[1:-1, 1:-1, 1:-1]

    win_sum1 = _window_sum_3d(img0, img1.shape, backend=backend)
    win_sum2 = _window_sum_3d(img0**2, img1.shape, backend=backend)

    template_mean = img1.mean()
    template_volume = np.prod(img1.shape)
    template_ssd = backend.sum((img1 - template_mean) ** 2)

    var = (win_sum2 - win_sum1**2 / template_volume) * template_ssd

    # zero division happens when perfectly matched
    response = backend.zeros(corr.shape, dtype=corr.dtype)
    mask = var > 0
    response[mask] = (corr - win_sum1 * template_mean)[mask] / _safe_sqrt(
        var, fill=np.inf, backend=backend
    )[mask]
    return response


def zncc_landscape_with_crop(
    img0: AnyArray[np.float32],
    img1: AnyArray[np.float32],
    max_shifts: tuple[float, ...],
    backend: Backend,
) -> AnyArray[np.float32]:
    response = ncc_landscape(
        img0 - img0.mean(), img1 - img1.mean(), max_shifts, backend=backend
    )
    pad_width_eff = tuple(
        (s - int(m) * 2 - 1) // 2 for m, s in zip(max_shifts, response.shape)
    )
    sl_res = tuple(slice(w, -w, None) for w in pad_width_eff)
    return response[sl_res]


def subpixel_zncc(
    img0: AnyArray[np.float32],
    img1: AnyArray[np.float32],
    upsample_factor: int,
    max_shifts: pixel | tuple[pixel, ...],
    backend: Backend,
) -> tuple[NDArray[np.float32], float]:
    img0 -= img0.mean()
    img1 -= img1.mean()
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * img0.ndim
    response = ncc_landscape(img0, img1, max_shifts, backend=backend)
    pad_width_eff = tuple(
        (s - int(m) * 2 - 1) // 2 for m, s in zip(max_shifts, response.shape)
    )
    sl_res = tuple(slice(w, -w, None) for w in pad_width_eff)
    response_center = response[sl_res]
    maxima = backend.unravel_index(
        backend.argmax(response_center), response_center.shape
    )
    midpoints = backend.asarray(response_center.shape, dtype=np.int32) // 2

    if upsample_factor > 1:
        coords = _create_mesh(
            upsample_factor,
            maxima,
            max_shifts,
            midpoints.astype(np.float32),
            pad_width_eff,
            backend,
        )
        local_response = backend.map_coordinates(
            response, coords, order=3, mode="constant", cval=-1.0, prefilter=True
        )
        local_maxima = backend.unravel_index(
            np.argmax(local_response), local_response.shape
        )
        zncc = local_response[local_maxima]
        shifts = (
            backend.asnumpy(maxima)
            - midpoints
            + backend.asnumpy(local_maxima) / upsample_factor
            - 1
        )
    else:
        zncc = response[maxima]
        shifts = backend.asnumpy(maxima) - midpoints

    return shifts, zncc


def _window_sum_2d(
    image: AnyArray[np.float32],
    window_shape: tuple[int, int, int],
    backend: Backend,
) -> AnyArray[np.float32]:
    window_sum = np.cumsum(image, axis=0)
    window_sum = window_sum[window_shape[0] : -1] - window_sum[: -window_shape[0] - 1]
    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (
        window_sum[:, window_shape[1] : -1] - window_sum[:, : -window_shape[1] - 1]
    )

    return window_sum


def _window_sum_3d(
    image: AnyArray[np.float32],
    window_shape: tuple[int, int, int],
    backend: Backend,
) -> AnyArray[np.float32]:
    window_sum = _window_sum_2d(image, window_shape, backend=backend)
    window_sum = backend.cumsum(window_sum, axis=2)
    window_sum = (
        window_sum[:, :, window_shape[2] : -1]
        - window_sum[:, :, : -window_shape[2] - 1]
    )

    return window_sum


def _safe_sqrt(a: AnyArray[np.float32], fill: float, backend: Backend):
    out = backend.full(a.shape, fill, dtype=np.float32)
    out = backend.zeros(a.shape, dtype=a.dtype)
    mask = a > 0
    out[mask] = backend.sqrt(a[mask])
    return out


@lru_cache(maxsize=12)
def _get_padding_width(max_shifts: tuple[int, ...]) -> list[tuple[int, ...]]:
    pad_width: list[tuple[int, ...]] = []
    for w in max_shifts:
        w_int = int(np.ceil(w + 3))
        pad_width.append((w_int,) * 2)

    return pad_width


def _create_mesh(
    upsample_factor: int,
    maxima: Sequence[np.intp],
    max_shifts: Sequence[pixel],
    midpoints: NDArray[np.float32],
    pad_width_eff: Sequence[pixel],
    backend: Backend,
):
    shifts = backend.array(maxima, dtype=np.float32) - midpoints
    _max_shifts = backend.array(max_shifts, dtype=np.float32)  # type: ignore
    left = -shifts - _max_shifts
    right = -shifts + _max_shifts
    local_shifts = tuple(
        [
            int(backend._xp_.round(max(shiftl, -1) * upsample_factor)),
            int(backend._xp_.round(min(shiftr, 1) * upsample_factor)),
        ]
        for shiftl, shiftr in zip(left, right)
    )
    mesh = backend.meshgrid(
        *[
            np.arange(s0, s1 + 1) / upsample_factor + m + w
            for (s0, s1), m, w in zip(local_shifts, maxima, pad_width_eff)
        ],
        indexing="ij",
    )
    return backend.stack(mesh, axis=0)


def fftconvolve(
    in1: AnyArray[np.float32],
    in2: AnyArray[np.float32],
    backend: Backend,
):
    s1 = in1.shape
    s2 = in2.shape

    # shape = in1.shape
    shape = [s1[i] + s2[i] - 1 for i in range(in1.ndim)]

    # convolve in the frequency domain
    fshape = [next_fast_len(shape[a], False) for a in range(in1.ndim)]

    sp1 = backend.rfftn(in1, fshape)
    sp2 = backend.rfftn(in2, fshape)
    ret = backend.irfftn(sp1 * sp2, fshape)

    fslice = tuple([slice(sz) for sz in shape])
    ret = ret[fslice]

    return _apply_conv_mode(ret, s1, s2)


def _apply_conv_mode(ret: AnyArray[np.float32], s1, s2):
    shape_valid = np.asarray([s1[a] - s2[a] + 1 for a in range(ret.ndim)])
    currshape = np.array(ret.shape)
    startind = (currshape - shape_valid) // 2
    endind = startind + shape_valid
    sl = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return ret[tuple(sl)]
