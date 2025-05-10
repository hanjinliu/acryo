from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from acryo._typed_scipy import fftn, ifftn
from acryo._types import nm


@dataclass
class CTFModel:
    """A model for Contrast Transfer Function.

    Attributes
    ----------
    spherical_aberration : float or callable
        Spherical aberration in mm.
    defocus : float or callable
        Defocus in μm.
    wave_length : float or callable
        Wave length in angstrom.
    bfactor : float
        B-factor.
    """

    spherical_aberration: float
    wave_length: float
    defocus: float = -1.0
    bfactor: float = 0.0

    @classmethod
    def from_kv(
        cls,
        kv: float,
        spherical_aberration: float,
        defocus: float = -1.0,
        bfactor: float = 0.0,
    ) -> CTFModel:
        wave_length = _voltage_to_wave_length(kv)
        return cls(
            spherical_aberration=spherical_aberration,
            wave_length=wave_length,
            defocus=defocus,
            bfactor=bfactor,
        )

    def simulate_image(
        self,
        shape: tuple[int, int],
        scale: nm,
    ) -> NDArray[np.floating]:
        """Simulate the CTF of the given shape."""
        yfreq = np.fft.fftfreq(shape[0], scale)
        xfreq = np.fft.fftfreq(shape[1], scale)
        yfreq, xfreq = np.meshgrid(yfreq, xfreq, indexing="ij")
        freq = np.sqrt(yfreq**2 + xfreq**2)
        return self.simulate(freq)

    def filter_apply_ctf(
        self,
        img: NDArray[np.floating],
        scale: nm,
    ) -> NDArray[np.floating]:
        """Apply the CTF to the image."""
        ctf = self.simulate_image(img.shape[-2:], scale)
        img_ft = fftn(img, axes=(-2, -1))
        return ifftn(_multiply_multi(ctf, img_ft), axes=(-2, -1)).real

    def filter_phase_flip(
        self,
        img: NDArray[np.floating],
        scale: nm,
    ) -> NDArray[np.floating]:
        """Flip the phase of the image."""
        ctf = self.simulate_image(img.shape[-2:], scale)
        img_ft = fftn(img, axes=(-2, -1))
        return ifftn(_multiply_multi(np.sign(ctf), img_ft), axes=(-2, -1)).real

    def filter_apply_phase_flip(
        self,
        img: NDArray[np.floating],
        scale: nm,
    ) -> NDArray[np.floating]:
        """Apply CTF to the image and flip the phase."""
        ctf = self.simulate_image(img.shape[-2:], scale)
        img_ft = fftn(img, axes=(-2, -1))
        return ifftn(_multiply_multi(np.sign(ctf) * ctf, img_ft), axes=(-2, -1)).real

    def filter_apply_phase_amplitude_correction(
        self,
        img: NDArray[np.floating],
        scale: nm,
        cutoff_amplitude: float = 1e-3,
    ) -> NDArray[np.floating]:
        """Apply amplitude correction to the image."""
        ctf = self.simulate_image(img.shape[-2:], scale)
        img_ft = fftn(img, axes=(-2, -1))
        zero_div_mask = ctf < cutoff_amplitude
        out = np.zeros_like(img_ft)
        out[..., ~zero_div_mask] = img_ft[..., ~zero_div_mask] / ctf[~zero_div_mask]
        out[..., zero_div_mask] = img_ft[..., zero_div_mask]
        return ifftn(out, axes=(-2, -1)).real

    def simulate(self, freq):
        f2 = freq**2
        cs = self.spherical_aberration * 1e6
        defocus = self.defocus * 1e3
        lmd = self.wave_length / 10
        wave_aberration = np.pi * lmd * defocus * f2 - np.pi / 2 * cs * lmd**3 * f2**2
        return np.sin(wave_aberration) * np.exp(-self.bfactor * f2 / 4)


def _voltage_to_wave_length(kv: float) -> float:
    """Convert kV to wave length in angstrom."""
    # energy of an electron is sum of relativistic momentum and kinetic energy
    return np.sqrt(0.1504 / (kv + 0.978e-3 * kv**2))


def _multiply_multi(
    a: NDArray[np.floating], b: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Multiply two arrays considering their dimensions."""
    if b.ndim == a.ndim:
        return a * b
    newaxis = (np.newaxis,) * (b.ndim - a.ndim)
    return a[newaxis] * b
