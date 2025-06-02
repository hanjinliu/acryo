from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

import scipy.fft

if TYPE_CHECKING:
    from acryo.ctf import CTFModel

# Modified from IsoNet https://github.com/IsoNet-cryoET/IsoNet (MIT License)


def wiener_deconv(
    vol: NDArray[np.float32],
    scale: float,
    ctf_model: CTFModel,
    snr_falloff: float = 1.0,
    deconv_strength: float = 1.0,
    highpass_nyquist: float = 0.02,
    phaseflipped: bool = False,
):
    data = np.linspace(0, 1, 2048)
    highpass = np.minimum(np.ones(data.size), data / highpass_nyquist) * np.pi
    highpass = 1 - np.cos(highpass)
    eps = 1e-6
    snr = (
        np.exp(-data * snr_falloff * 10 / scale) * (10**deconv_strength) * highpass
        + eps
    )

    ctf = ctf_model.simulate(np.fft.fftfreq(2048, scale))
    if phaseflipped:
        ctf = np.abs(ctf)

    wiener = ctf / (ctf * ctf + 1 / snr)

    s1 = -int(vol.shape[1] / 2)
    f1 = s1 + vol.shape[1] - 1
    m1 = np.arange(s1, f1 + 1)

    s2 = -int(vol.shape[0] / 2)
    f2 = s2 + vol.shape[0] - 1
    m2 = np.arange(s2, f2 + 1)

    s3 = -int(vol.shape[2] / 2)
    f3 = s3 + vol.shape[2] - 1
    m3 = np.arange(s3, f3 + 1)

    x, y, z = np.meshgrid(m1, m2, m3)
    x = x.astype(np.float32) / np.abs(s1)
    y = y.astype(np.float32) / np.abs(s2)
    z = z.astype(np.float32) / np.maximum(1, np.abs(s3))
    r = np.sqrt(x**2 + y**2 + z**2)
    del x, y, z
    r = np.minimum(1, r)
    r = np.fft.ifftshift(r)

    ramp = np.interp(r, data, wiener).astype(np.float32)
    del r

    deconv = np.real(
        scipy.fft.ifftn(scipy.fft.fftn(vol, overwrite_x=True) * ramp, overwrite_x=True)
    )
    std_deconv = np.std(deconv)
    std_vol = np.std(vol)
    ave_vol = np.average(vol)
    del vol, ramp

    deconv /= std_deconv
    deconv *= std_vol
    deconv += ave_vol
    return deconv
