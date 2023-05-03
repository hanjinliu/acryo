from __future__ import annotations
from typing import TypeVar
import numpy as np
from numpy.typing import NDArray

from acryo.backend import Backend, AnyArray

# Modified from skimage/registration/_phase_cross_correlation.py


def pcc_landscape(
    f0: AnyArray[np.complex64],
    f1: AnyArray[np.complex64],
    max_shifts: tuple[float, ...],
    backend: Backend,
):
    product = f0 * f1.conj()
    power = _abs2(backend.ifftn(product))
    power = backend.fftshift(power)
    centers = tuple(s // 2 for s in power.shape)
    slices = tuple(
        slice(max(c - int(shiftl), 0), min(c + int(shiftr) + 1, s), None)
        for c, shiftl, shiftr, s in zip(centers, max_shifts, max_shifts, power.shape)
    )
    power = power[slices]
    return power


def subpixel_pcc(
    f0: AnyArray[np.complex64],
    f1: AnyArray[np.complex64],
    upsample_factor: int,
    max_shifts: tuple[float, ...] | NDArray[np.number],
    backend: Backend,
) -> tuple[NDArray[np.float32], float]:
    if isinstance(max_shifts, (int, float)):
        max_shifts = (max_shifts,) * f0.ndim
    product = f0 * f1.conj()
    power = _abs2(backend.ifftn(product))
    if max_shifts is not None:
        max_shifts = backend.asarray(max_shifts)
        power = crop_by_max_shifts(power, max_shifts, max_shifts, backend)

    maxima = backend.unravel_index(backend.argmax(power), power.shape)
    midpoints = backend.array(
        [backend._xp_.fix(axis_size / 2) for axis_size in power.shape]
    )

    shifts = backend.asarray(maxima, dtype=np.float32)
    shifts[shifts > midpoints] -= backend.array(power.shape)[shifts > midpoints]
    # Initial shift estimate in upsampled grid
    shifts = backend._xp_.fix(shifts * upsample_factor) / upsample_factor
    if upsample_factor > 1:
        upsampled_region_size = backend._xp_.ceil(upsample_factor * 1.5)
        # Center of output array at dftshift + 1
        dftshift = backend._xp_.fix(upsampled_region_size / 2.0)
        # Matrix multiply DFT around the current shift estimate
        sample_region_offset = dftshift - shifts * upsample_factor
        # Locate maximum and map back to original pixel grid
        power = _abs2(
            _upsampled_dft(
                product.conj(),
                upsampled_region_size,
                upsample_factor,
                sample_region_offset,
                backend,
            )
        )

        if max_shifts is not None:
            _upsampled_left_shifts = (shifts + max_shifts) * upsample_factor
            _upsampled_right_shifts = (max_shifts - shifts) * upsample_factor
            power = crop_by_max_shifts(
                power, _upsampled_left_shifts, _upsampled_right_shifts, backend
            )

        maxima = backend.unravel_index(backend.argmax(power), power.shape)
        maxima = backend.asarray(maxima, dtype=np.float32) - dftshift
        shifts = shifts + maxima / upsample_factor
        pcc = backend._xp_.sqrt(
            power[tuple(int(backend._xp_.round(m)) for m in maxima)]
        )
    else:
        pcc = backend._xp_.sqrt(power[maxima])
    return shifts, pcc


_DType = TypeVar("_DType", bound=np.number)


def _upsampled_dft(
    data: AnyArray[_DType],
    upsampled_region_size: AnyArray[np.integer],
    upsample_factor: int,
    axis_offsets: AnyArray[np.float32],
    backend: Backend,
) -> AnyArray[_DType]:
    # if people pass in an integer, expand it to a list of equal-sized sections
    upsampled_region_sizes = [upsampled_region_size] * data.ndim

    dim_properties = list(zip(data.shape, upsampled_region_sizes, axis_offsets))

    for n_items, ups_size, ax_offset in dim_properties[::-1]:
        kernel = (backend._xp_.arange(ups_size, dtype=np.float32) - ax_offset)[
            :, np.newaxis
        ] * backend.fftfreq(n_items, upsample_factor).astype(np.float32)
        kernel = backend._xp_.exp(-2j * np.pi * kernel)

        data = backend._xp_.tensordot(kernel, data, axes=(1, -1))  # type: ignore
    return data


def _abs2(a: AnyArray[np.complex64]) -> AnyArray[np.float32]:
    return a.real**2 + a.imag**2


def crop_by_max_shifts(
    power: AnyArray[_DType],
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    backend: Backend,
) -> AnyArray[_DType]:
    shifted_power = backend.fftshift(power)
    centers = tuple(s // 2 for s in power.shape)
    slices = tuple(
        slice(max(c - int(shiftl), 0), min(c + int(shiftr) + 1, s), None)
        for c, shiftl, shiftr, s in zip(centers, left, right, power.shape)
    )
    return backend.ifftshift(shifted_power[slices])
