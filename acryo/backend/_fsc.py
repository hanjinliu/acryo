from __future__ import annotations
from functools import lru_cache
import numpy as np

from acryo.backend import Backend, AnyArray
from acryo._types import pixel
from ._upsample import upsample


def fsc_landscape(
    ft0: AnyArray[np.complex64],
    ft1: AnyArray[np.complex64],
    max_shifts: tuple[pixel, pixel, pixel],
    backend: Backend,
) -> AnyArray[np.float32]:
    shape = ft0.shape
    labels = _get_radial_label(shape, backend)
    nlabels = labels.max()
    index = backend.arange(0, nlabels)

    pw0 = ft0.real**2 + ft0.imag**2
    pw1 = ft1.real**2 + ft1.imag**2
    sigma_product = backend.sqrt(
        backend.sum_labels(pw0, labels=labels, index=index)
        * backend.sum_labels(pw1, labels=labels, index=index)
    )
    out_shape = tuple(int(np.ceil(m)) * 2 + 1 for m in max_shifts)
    mesh = backend.meshgrid(
        *[backend.fftshift(backend.fftfreq(s, d=1.0)) for s in shape], indexing="ij"
    )
    out = backend.zeros(out_shape)

    phase_z = _get_phase_1d(mesh[0], out_shape[0], backend)
    phase_y = _get_phase_1d(mesh[1], out_shape[1], backend)
    phase_x = _get_phase_1d(mesh[2], out_shape[2], backend)
    for iz, phiz in enumerate(phase_z):
        ft0_shifted_z = ft0 * phiz
        for iy, phiy in enumerate(phase_y):
            ft0_shifted_yz = ft0_shifted_z * phiy
            for ix, phix in enumerate(phase_x):
                ft0_shifted = ft0_shifted_yz * phix
                cov = ft0_shifted.real * ft1.real + ft0_shifted.imag * ft1.imag
                fsc = (
                    backend.sum_labels(cov, labels=labels, index=index) / sigma_product
                )
                out[iz, iy, ix] = float(fsc.mean())
    return out


def subpixel_fsc(
    ft0: AnyArray[np.complex64],
    ft1: AnyArray[np.complex64],
    max_shifts: tuple[pixel, pixel, pixel],
    backend: Backend,
) -> tuple[AnyArray[np.float32], float]:
    out = fsc_landscape(ft0, ft1, max_shifts, backend=backend)
    return upsample(out, out, max_shifts, (0, 0, 0), backend=backend)


@lru_cache(maxsize=12)
def _get_radial_label(
    shape: tuple[int, int, int], backend: Backend
) -> AnyArray[np.uint16]:
    freqs = backend.meshgrid(
        *[backend.fftshift(backend.fftfreq(s, d=1.0)) for s in shape], indexing="ij"
    )  # type: ignore
    r = backend.sqrt(sum(f**2 for f in freqs))
    dfreq = 1 / min(shape)
    return (r / dfreq).astype(np.uint16)


def _get_phase_1d(
    mesh: AnyArray[np.float32], size: int, backend: Backend
) -> list[AnyArray[np.complex64]]:
    s = size // 2
    rng = range(-s, s + 1)
    return [backend.exp(2j * np.pi * x0 * mesh) for x0 in rng]
