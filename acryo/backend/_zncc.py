from __future__ import annotations
from functools import lru_cache
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len

from acryo._types import pixel
from acryo.backend import Backend, AnyArray

# Normalized cross correlation
UPSAMPLE = 20


def ncc_landscape(
    img0: AnyArray[np.float32],
    img1: AnyArray[np.float32],
    max_shifts: tuple[float, ...],
    backend: Backend,
) -> AnyArray[np.float32]:
    if max_shifts is not None:
        max_shifts = tuple(max_shifts)
    pad_width = _get_padding_width(max_shifts)
    padimg = backend.pad(img0, pad_width=pad_width, mode="constant", constant_values=0)
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
    template_volume = np.prod(img1.shape, dtype=np.float32)  # type: ignore
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
    max_shifts: pixel | tuple[pixel, ...],
    backend: Backend,
) -> tuple[NDArray[np.float32], float]:
    """Get the optimal subpixel shift and the score."""
    img0 = img0 - img0.mean()
    img1 = img1 - img1.mean()
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * img0.ndim
    response = ncc_landscape(img0, img1, max_shifts, backend=backend)
    pad_width_eff = tuple(
        (s - int(m) * 2 - 1) // 2 for m, s in zip(max_shifts, response.shape)
    )
    sl_res = tuple(slice(w, -w, None) for w in pad_width_eff)
    response_center = response[sl_res]
    maxima = backend.asnumpy(
        backend.unravel_index(backend.argmax(response_center), response_center.shape)
    )
    midpoints = np.asarray(response_center.shape, dtype=np.int32) // 2
    coords, local_offset = _create_mesh(
        maxima,
        max_shifts,
        midpoints.astype(np.float32),
        pad_width_eff,
        backend,
    )
    local_response = backend.map_coordinates(
        response, coords, order=3, mode="constant", cval=-1.0, prefilter=True
    )
    local_maxima = backend.asnumpy(
        backend.unravel_index(backend.argmax(local_response), local_response.shape)
    )
    zncc = backend.asnumpy(local_response[tuple(local_maxima)])
    loc_shift = local_maxima / UPSAMPLE + local_offset
    shifts = maxima - midpoints + loc_shift

    return shifts, zncc  # type: ignore


def _window_sum_2d(
    image: AnyArray[np.float32],
    window_shape: tuple[int, ...],
    backend: Backend,
) -> AnyArray[np.float32]:
    window_sum = backend.cumsum(image, axis=0)
    window_sum = window_sum[window_shape[0] : -1] - window_sum[: -window_shape[0] - 1]
    window_sum = backend.cumsum(window_sum, axis=1)
    window_sum = (
        window_sum[:, window_shape[1] : -1] - window_sum[:, : -window_shape[1] - 1]
    )

    return window_sum


def _window_sum_3d(
    image: AnyArray[np.float32],
    window_shape: tuple[int, ...],
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
def _get_padding_width(max_shifts: tuple[int, ...]) -> list[tuple[int, int]]:
    pad_width: list[tuple[int, int]] = []
    for w in max_shifts:
        w_int = int(np.ceil(w + 3))
        pad_width.append((w_int, w_int))

    return pad_width


def _create_mesh(
    maxima: NDArray[np.intp],
    max_shifts: Sequence[pixel],
    midpoints: NDArray[np.float32],
    pad_width_eff: Sequence[pixel],
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
    fshape: tuple[int, int, int] = tuple(
        next_fast_len(shape[a], False) for a in range(in1.ndim)  # type: ignore
    )

    sp1 = backend.rfftn(in1, fshape)
    sp2 = backend.rfftn(in2, fshape)
    ret = backend.irfftn(sp1 * sp2, fshape)

    fslice = tuple([slice(sz) for sz in shape])
    ret = ret[fslice]

    return _apply_conv_mode(ret, s1, s2)


def _apply_conv_mode(
    ret: AnyArray[np.float32], s1: tuple[int, ...], s2: tuple[int, ...]
):
    shape_valid = np.array([s1[a] - s2[a] + 1 for a in range(ret.ndim)], dtype=np.int32)
    currshape = np.array(ret.shape)
    startind = (currshape - shape_valid) // 2
    endind = startind + shape_valid
    sl = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return ret[tuple(sl)]
