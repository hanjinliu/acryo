from __future__ import annotations
import itertools
from functools import reduce, lru_cache
from typing import Callable, Sequence
import numpy as np
from numpy.typing import NDArray

from scipy.signal import fftconvolve
from scipy.spatial.transform import Rotation
from scipy import ndimage as ndi

from acryo._fft import rfftn, irfftn, fftn, ifftn
from acryo._types import Ranges, RangeLike, pixel
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
    from .._utils import compose_matrices

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


# lowpass filter
# Modified from skimage.filters._fft_based
def lowpass_filter_ft(
    img: NDArray[np.float32], cutoff: float, order: int = 2
) -> NDArray[np.complex64]:
    if cutoff >= 0.5 * np.sqrt(img.ndim) or cutoff <= 0:
        return fftn(img)
    weight = _get_ND_butterworth_filter(
        img.shape,
        cutoff,
        order,
        high_pass=False,
        real=False,
    )
    return weight * fftn(img)


def lowpass_filter(
    img: NDArray[np.float32], cutoff: float, order: int = 2
) -> NDArray[np.float32]:
    if cutoff >= 0.5 * np.sqrt(img.ndim) or cutoff <= 0:
        return img
    weight = _get_ND_butterworth_filter(
        img.shape,
        cutoff,
        order,
        high_pass=False,
        real=True,
    )
    out: np.ndarray = irfftn(weight * rfftn(img))
    return out.real


@lru_cache(maxsize=4)
def _get_ND_butterworth_filter(
    shape: tuple[int, ...],
    cutoff: float,
    order: int,
    high_pass: bool,
    real: bool,
):
    ranges = []
    for d in shape:
        axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1, dtype=np.float32) / (
            d * cutoff
        )
        ranges.append(np.fft.ifftshift(axis**2))
    if real:
        limit = shape[-1] // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    q2 = reduce(np.add, np.meshgrid(*ranges, indexing="ij", sparse=True))
    wfilt = 1 / (1 + q2**order)
    if high_pass:
        wfilt = 1 - wfilt
    return wfilt


# cross correlation
# Modified from skimage/registration/_phase_cross_correlation.py


def subpixel_pcc(
    f0: NDArray[np.complex64],
    f1: NDArray[np.complex64],
    upsample_factor: int,
    max_shifts: tuple[float, ...] | NDArray[np.number] | None = None,
) -> tuple[NDArray[np.float32], float]:
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * f0.ndim
    product = f0 * f1.conj()
    power = abs2(ifftn(product))
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
        power = abs2(
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


def _upsampled_dft(
    data: NDArray[np.number],
    upsampled_region_size: NDArray[np.integer],
    upsample_factor: int,
    axis_offsets: NDArray[np.float32],
) -> NDArray[np.number]:
    # if people pass in an integer, expand it to a list of equal-sized sections
    upsampled_region_sizes = [upsampled_region_size] * data.ndim

    dim_properties = list(zip(data.shape, upsampled_region_sizes, axis_offsets))

    for (n_items, ups_size, ax_offset) in dim_properties[::-1]:
        kernel = (np.arange(ups_size) - ax_offset)[  # type: ignore
            :, np.newaxis
        ] * np.fft.fftfreq(  # type: ignore
            n_items, upsample_factor
        )
        kernel = np.exp(-2j * np.pi * kernel)

        data = np.tensordot(kernel, data, axes=(1, -1))  # type: ignore
    return data


def abs2(a: NDArray[np.number]) -> NDArray[np.number]:
    return a.real**2 + a.imag**2


def crop_by_max_shifts(power: np.ndarray, left, right):
    shifted_power = np.fft.fftshift(power)
    centers = tuple(s // 2 for s in power.shape)
    slices = tuple(
        slice(max(c - int(shiftl), 0), min(c + int(shiftr) + 1, s), None)
        for c, shiftl, shiftr, s in zip(centers, left, right, power.shape)
    )
    return np.fft.ifftshift(shifted_power[slices])


# Normalized cross correlation


def _draw_ncc_landscape_no_crop(
    img0: np.ndarray,
    img1: np.ndarray,
    max_shifts: tuple[float, ...] | None = None,
) -> np.ndarray:
    ndim = img1.ndim
    if max_shifts is not None:
        max_shifts = tuple(max_shifts)
    pad_width, sl = _get_padding_params(img0.shape, img1.shape, max_shifts)
    padimg = np.pad(img0[sl], pad_width=pad_width, mode="constant", constant_values=0)  # type: ignore

    corr = fftconvolve(padimg, img1[(slice(None, None, -1),) * ndim], mode="valid")[
        (slice(1, -1, None),) * ndim
    ]

    _win_sum = _window_sum_2d if ndim == 2 else _window_sum_3d
    win_sum1 = _win_sum(padimg, img1.shape)
    win_sum2 = _win_sum(padimg**2, img1.shape)

    template_mean = np.mean(img1)
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


def subpixel_zncc(
    img0: NDArray[np.float32],
    img1: NDArray[np.float32],
    upsample_factor: int,
    max_shifts: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, float]:
    img0, img1 = img0.astype(np.float32), img1.astype(np.float32)
    img0 -= img0.mean()
    img1 -= img1.mean()
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * img0.ndim
    response = _draw_ncc_landscape_no_crop(img0, img1, max_shifts)
    if max_shifts is None:
        pad_width_eff = (3,) * img1.ndim
    else:
        pad_width_eff = tuple(
            (s - int(m) * 2 - 1) // 2 for m, s in zip(max_shifts, response.shape)
        )
    sl_res = tuple(slice(w, -w, None) for w in pad_width_eff)
    response_center: np.ndarray = response[sl_res]
    maxima = np.unravel_index(np.argmax(response_center), response_center.shape)
    midpoints = np.asarray(response_center.shape) // 2

    if upsample_factor > 1:
        coords = _create_mesh(
            upsample_factor,
            maxima,  # type: ignore
            max_shifts,  # type: ignore
            midpoints,  # type: ignore
            pad_width_eff,  # type: ignore
        )
        local_response: np.ndarray = ndi.map_coordinates(
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


# Identical to skimage.feature.template, but compatible between numpy and cupy.
def _window_sum_2d(image, window_shape):
    window_sum = np.cumsum(image, axis=0)
    window_sum = window_sum[window_shape[0] : -1] - window_sum[: -window_shape[0] - 1]  # type: ignore
    window_sum = np.cumsum(window_sum, axis=1)
    window_sum = (
        window_sum[:, window_shape[1] : -1] - window_sum[:, : -window_shape[1] - 1]
    )  # type: ignore

    return window_sum


def _window_sum_3d(image, window_shape):
    window_sum = _window_sum_2d(image, window_shape)
    window_sum = np.cumsum(window_sum, axis=2)
    window_sum = (
        window_sum[:, :, window_shape[2] : -1]
        - window_sum[:, :, : -window_shape[2] - 1]
    )  # type: ignore

    return window_sum


def _safe_sqrt(a: np.ndarray, fill: float = 0.0):
    out = np.full(a.shape, fill, dtype=np.float32)
    out = np.zeros_like(a)
    mask = a > 0
    out[mask] = np.sqrt(a[mask])
    return out


@lru_cache(maxsize=12)
def _get_padding_params(
    shape0: tuple[int, ...],
    shape1: tuple[int, ...],
    max_shifts: tuple[int, ...] | None,
) -> tuple[list[tuple[int, ...]], tuple[slice, ...] | slice]:
    if max_shifts is None:
        pad_width = [(w, w) for w in shape1]
        sl = slice(None)
    else:
        pad_width: list[tuple[int, ...]] = []
        _sl: list[slice] = []
        for w, s0, s1 in zip(max_shifts, shape0, shape1):
            w_int = int(np.ceil(w + 3 - (s0 - s1) / 2))
            if w_int >= 0:
                pad_width.append((w_int,) * 2)
                _sl.append(slice(None))
            else:
                pad_width.append((0,) * 2)
                _sl.append(slice(-w_int, w_int, None))
        sl = tuple(_sl)

    return pad_width, sl


def _create_mesh(
    upsample_factor: int,
    maxima: Sequence[pixel],
    max_shifts: Sequence[pixel] | None,
    midpoints: Sequence[pixel],
    pad_width_eff: Sequence[pixel],
):
    if max_shifts is not None:
        shifts = np.array(maxima, dtype=np.float32) - np.array(
            midpoints, dtype=np.float32
        )  # type: ignore
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
    else:
        local_shifts = ([-upsample_factor, upsample_factor],) * len(maxima)
    mesh = np.meshgrid(
        *[
            np.arange(s0, s1 + 1) / upsample_factor + m + w
            for (s0, s1), m, w in zip(local_shifts, maxima, pad_width_eff)
        ],
        indexing="ij",
    )
    return np.stack(mesh, axis=0)
