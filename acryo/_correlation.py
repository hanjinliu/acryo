from __future__ import annotations
from functools import lru_cache
from typing import Sequence, TypeVar
import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len

from acryo._typed_scipy import rfftn, irfftn, ifftn, map_coordinates
from acryo._types import pixel

# cross correlation
# Modified from skimage/registration/_phase_cross_correlation.py


def pcc_landscape(
    f0: NDArray[np.complex64],
    f1: NDArray[np.complex64],
    max_shifts: tuple[float, ...] | None = None,
):
    product = f0 * f1.conj()
    power = _abs2(ifftn(product))
    power = np.fft.fftshift(power)
    if max_shifts is not None:
        centers = tuple(s // 2 for s in power.shape)
        slices = tuple(
            slice(max(c - int(shiftl), 0), min(c + int(shiftr) + 1, s), None)
            for c, shiftl, shiftr, s in zip(
                centers, max_shifts, max_shifts, power.shape
            )
        )
        power = power[slices]
    return power


def subpixel_pcc(
    f0: NDArray[np.complex64],
    f1: NDArray[np.complex64],
    upsample_factor: int,
    max_shifts: tuple[float, ...] | NDArray[np.number] | None = None,
) -> tuple[NDArray[np.float32], float]:
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * f0.ndim
    product = f0 * f1.conj()
    power = _abs2(ifftn(product))
    if max_shifts is not None:
        max_shifts = np.asarray(max_shifts)
        power = crop_by_max_shifts(power, max_shifts, max_shifts)

    maxima = np.unravel_index(np.argmax(power), power.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in power.shape])

    shifts = np.asarray(maxima, dtype=np.float32)
    shifts[shifts > midpoints] -= np.array(power.shape)[shifts > midpoints]
    # Initial shift estimate in upsampled grid
    shifts = np.fix(shifts * upsample_factor) / upsample_factor
    if upsample_factor > 1:
        upsampled_region_size = np.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = np.fix(upsampled_region_size / 2.0)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        # Locate maximum and map back to original pixel grid
        power = _abs2(
            _upsampled_dft(
                product.conj(),
                upsampled_region_size,
                upsample_factor,
                sample_region_offset,
            )
        )

        if max_shifts is not None:
            _upsampled_left_shifts = (shifts + max_shifts) * upsample_factor
            _upsampled_right_shifts = (max_shifts - shifts) * upsample_factor
            power = crop_by_max_shifts(
                power, _upsampled_left_shifts, _upsampled_right_shifts
            )

        maxima = np.unravel_index(np.argmax(power), power.shape)
        maxima = np.asarray(maxima, dtype=np.float32) - dftshift
        shifts = shifts + maxima / upsample_factor
        pcc = np.sqrt(power[tuple(int(np.round(m)) for m in maxima)])
    else:
        pcc = np.sqrt(power[maxima])
    return shifts, pcc


_DType = TypeVar("_DType", bound=np.number)


def _upsampled_dft(
    data: NDArray[_DType],
    upsampled_region_size: NDArray[np.integer],
    upsample_factor: int,
    axis_offsets: NDArray[np.float32],
) -> NDArray[_DType]:
    # if people pass in an integer, expand it to a list of equal-sized sections
    upsampled_region_sizes = [upsampled_region_size] * data.ndim

    dim_properties = list(zip(data.shape, upsampled_region_sizes, axis_offsets))

    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (np.arange(ups_size, dtype=np.float32) - ax_offset)[
            :, np.newaxis
        ] * np.fft.fftfreq(n_items, upsample_factor).astype(np.float32)
        kernel = np.exp(-2j * np.pi * kernel)

        data = np.tensordot(kernel, data, axes=(1, -1))  # type: ignore
    return data


def _abs2(a: NDArray[np.complex64]) -> NDArray[np.float32]:
    return a.real**2 + a.imag**2


def crop_by_max_shifts(power: NDArray[_DType], left, right) -> NDArray[_DType]:
    shifted_power = np.fft.fftshift(power)
    centers = tuple(s // 2 for s in power.shape)
    slices = tuple(
        slice(max(c - int(shiftl), 0), min(c + int(shiftr) + 1, s), None)
        for c, shiftl, shiftr, s in zip(centers, left, right, power.shape)
    )
    return np.fft.ifftshift(shifted_power[slices])


# Normalized cross correlation


def ncc_landscape(
    img0: NDArray[np.float32],
    img1: NDArray[np.float32],
    max_shifts: tuple[float, ...],
) -> NDArray[np.float32]:
    if max_shifts is not None:
        max_shifts = tuple(max_shifts)
    pad_width = _get_padding_width(max_shifts)
    padimg = np.pad(img0, pad_width=pad_width, mode="constant", constant_values=0)

    corr = fftconvolve(padimg, img1[::-1, ::-1, ::-1])[1:-1, 1:-1, 1:-1]

    win_sum1 = _window_sum_3d(padimg, img1.shape)
    win_sum2 = _window_sum_3d(padimg**2, img1.shape)

    template_mean = np.mean(img1, dtype=np.float32)
    template_volume = np.prod(img1.shape)
    template_ssd = np.sum((img1 - template_mean) ** 2)

    var = (win_sum2 - win_sum1**2 / template_volume) * template_ssd

    # zero division happens when perfectly matched
    response = np.zeros_like(corr)
    mask = var > 0
    response[mask] = (corr - win_sum1 * template_mean)[mask] / _safe_sqrt(
        var, fill=np.inf
    )[mask]
    return response


def ncc_landscape_no_pad(
    img: NDArray[np.float32],
    template: NDArray[np.float32],
) -> NDArray[np.float32]:
    corr = fftconvolve(img, template[::-1, ::-1, ::-1])[1:-1, 1:-1, 1:-1]

    win_sum1 = _window_sum_3d(img, template.shape)
    win_sum2 = _window_sum_3d(img**2, template.shape)

    template_mean = np.mean(template)
    template_volume = np.prod(template.shape)
    template_ssd = np.sum((template - template_mean) ** 2)

    var = (win_sum2 - win_sum1**2 / template_volume) * template_ssd

    # zero division happens when perfectly matched
    response = np.zeros_like(corr)
    mask = var > 0
    response[mask] = (corr - win_sum1 * template_mean)[mask] / _safe_sqrt(
        var, fill=np.inf
    )[mask]
    return response


def zncc_landscape_with_crop(
    img0: NDArray[np.float32],
    img1: NDArray[np.float32],
    max_shifts: tuple[float, ...],
):
    response = ncc_landscape(img0 - img0.mean(), img1 - img1.mean(), max_shifts)
    pad_width_eff = tuple(
        (s - int(m) * 2 - 1) // 2 for m, s in zip(max_shifts, response.shape)
    )
    sl_res = tuple(slice(w, -w, None) for w in pad_width_eff)
    return response[sl_res]


def subpixel_zncc(
    img0: NDArray[np.float32],
    img1: NDArray[np.float32],
    upsample_factor: int,
    max_shifts: pixel | tuple[pixel, ...],
) -> tuple[np.ndarray, float]:
    img0 -= img0.mean()
    img1 -= img1.mean()
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * img0.ndim
    response = ncc_landscape(img0, img1, max_shifts)
    pad_width_eff = tuple(
        (s - int(m) * 2 - 1) // 2 for m, s in zip(max_shifts, response.shape)
    )
    sl_res = tuple(slice(w, -w, None) for w in pad_width_eff)
    response_center = response[sl_res]
    maxima = np.unravel_index(np.argmax(response_center), response_center.shape)
    midpoints = np.asarray(response_center.shape, dtype=np.int32) // 2

    if upsample_factor > 1:
        coords = _create_mesh(
            upsample_factor,
            maxima,
            max_shifts,
            midpoints.astype(np.float32),
            pad_width_eff,
        )
        local_response: np.ndarray = map_coordinates(
            response, coords, order=3, mode="constant", cval=-1.0, prefilter=True
        )
        local_maxima = np.unravel_index(np.argmax(local_response), local_response.shape)
        zncc = local_response[local_maxima]
        shifts = (
            np.array(maxima) - midpoints + np.array(local_maxima) / upsample_factor - 1
        )
    else:
        zncc = response[maxima]
        shifts = np.array(maxima) - midpoints

    return np.asarray(shifts, dtype=np.float32), zncc


def _window_sum_2d(image: NDArray[np.float32], window_shape: tuple[int, int, int]):
    window_sum = np.cumsum(image, axis=0)
    window_sum = window_sum[window_shape[0] : -1] - window_sum[: -window_shape[0] - 1]
    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (
        window_sum[:, window_shape[1] : -1] - window_sum[:, : -window_shape[1] - 1]
    )

    return window_sum


def _window_sum_3d(image: NDArray[np.float32], window_shape: tuple[int, int, int]):
    window_sum = _window_sum_2d(image, window_shape)
    window_sum = np.cumsum(window_sum, axis=2)
    window_sum = (
        window_sum[:, :, window_shape[2] : -1]
        - window_sum[:, :, : -window_shape[2] - 1]
    )

    return window_sum


def _safe_sqrt(a: np.ndarray, fill: float = 0.0):
    out = np.full(a.shape, fill, dtype=np.float32)
    out = np.zeros_like(a)
    mask = a > 0
    out[mask] = np.sqrt(a[mask])
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
):
    shifts = np.array(maxima, dtype=np.float32) - midpoints
    _max_shifts = np.array(max_shifts, dtype=np.float32)  # type: ignore
    left = -shifts - _max_shifts
    right = -shifts + _max_shifts
    local_shifts = tuple(
        [
            int(np.round(max(shiftl, -1) * upsample_factor)),
            int(np.round(min(shiftr, 1) * upsample_factor)),
        ]
        for shiftl, shiftr in zip(left, right)
    )
    mesh = np.meshgrid(
        *[
            np.arange(s0, s1 + 1) / upsample_factor + m + w
            for (s0, s1), m, w in zip(local_shifts, maxima, pad_width_eff)
        ],
        indexing="ij",
    )
    return np.stack(mesh, axis=0)


def fftconvolve(in1: NDArray[np.float32], in2: NDArray[np.float32]):
    s1 = in1.shape
    s2 = in2.shape

    # shape = in1.shape
    shape = [s1[i] + s2[i] - 1 for i in range(in1.ndim)]

    ret = _freq_domain_conv(in1, in2, shape)

    return _apply_conv_mode(ret, s1, s2)


def _freq_domain_conv(in1: NDArray[np.float32], in2: NDArray[np.float32], shape):
    fshape = [next_fast_len(shape[a], False) for a in range(in1.ndim)]

    sp1 = rfftn(in1, fshape)
    sp2 = rfftn(in2, fshape)
    ret = irfftn(sp1 * sp2, fshape)

    fslice = tuple([slice(sz) for sz in shape])
    ret = ret[fslice]

    return ret


def _apply_conv_mode(ret: NDArray[np.float32], s1, s2):
    shape_valid = np.asarray([s1[a] - s2[a] + 1 for a in range(ret.ndim)])
    currshape = np.array(ret.shape)
    startind = (currshape - shape_valid) // 2
    endind = startind + shape_valid
    myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    return ret[tuple(myslice)]
