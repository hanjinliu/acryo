import numpy as np
import pytest
from acryo.ctf import CTFModel


def test_ctf():
    ctf = CTFModel.from_kv(300, 2.7, defocus=-2.1)
    img = ctf.simulate_image((64, 64), 0.2)
    assert img.shape == (64, 64)
    rng = np.random.default_rng(0)
    img = rng.random((72, 60))
    assert ctf.apply_ctf(img, scale=0.2).shape == (72, 60)
    assert ctf.phase_flip(img, scale=0.2).shape == (72, 60)


def _gaussian_blob(shape: tuple[int, ...], sigma: float = 3.0):
    coords = np.indices(shape) - np.array(shape).reshape(-1, *([1] * len(shape))) // 2
    return np.exp(-np.sum(coords**2, axis=0) / (2 * sigma**2)).astype(np.float32)


@pytest.mark.parametrize("defocus", [-1.0, -3.0, -5.0])
def test_ctf_preserves_contrast(defocus: float):
    ctf = CTFModel.from_kv(300, 2.7, defocus=defocus)
    # first lobe is positive under underfocus
    assert np.all(ctf.simulate(np.linspace(0.01, 0.2, 10)) > 0)

    blob = _gaussian_blob((64, 64))
    img = ctf.apply_ctf(blob, scale=1.0)
    assert img[32, 32] > 0
    assert ctf.phase_flip(img, scale=1.0)[32, 32] > 0


def _tom_ctf1d(pixelsize, voltage, cs, defocus, amplitude, length=2048):
    # Reference implementation from IsoNet (MIT License), in SI units. Phase shift
    # and B-factor are omitted.
    # https://github.com/IsoNet-cryoET/IsoNet/blob/master/util/deconvolution.py
    ny = 1 / pixelsize
    lambda1 = 12.2643247 / np.sqrt(voltage * (1.0 + voltage * 0.978466e-6)) * 1e-10
    lambda2 = lambda1 * 2
    points = np.arange(0, length) / (2 * length) * ny
    k2 = points**2
    term1 = lambda1**3 * cs * k2**2
    w = np.pi / 2 * (term1 + lambda2 * defocus * k2)
    acurve = np.cos(w) * amplitude
    pcurve = -np.sqrt(1 - amplitude**2) * np.sin(w)
    return points, lambda1, pcurve + acurve


@pytest.mark.parametrize("amplitude", [0.0, 0.07, 0.1, 1.0])
@pytest.mark.parametrize("defocus", [-1.0, -3.0])
def test_ctf_matches_isonet(amplitude: float, defocus: float):
    scale = 0.5
    # IsoNet takes positive values for underfocus and negates them before passing
    # to `tom_ctf1d`, which is the same as our sign convention.
    freq, wave_length, ctf_ref = _tom_ctf1d(
        scale * 1e-9, 300e3, 2.7e-3, defocus * 1e-6, amplitude
    )
    ctf = CTFModel(
        spherical_aberration=2.7,
        wave_length=wave_length * 1e10,
        defocus=defocus,
        amplitude=amplitude,
    )
    np.testing.assert_allclose(ctf.simulate(freq * 1e-9), ctf_ref, atol=1e-8)


def test_ctf_amplitude():
    for amplitude in [0.0, 0.07, 0.5]:
        ctf = CTFModel.from_kv(300, 2.7, defocus=-3.0, amplitude=amplitude)
        assert ctf.simulate(0.0) == pytest.approx(amplitude)
    with pytest.raises(ValueError):
        CTFModel.from_kv(300, 2.7, amplitude=-0.1)
    with pytest.raises(ValueError):
        CTFModel.from_kv(300, 2.7, amplitude=1.1)


def test_ctf_cs_term():
    # spherical aberration compensates the phase shift of underfocus
    freq = np.linspace(0.01, 3.0, 300)
    ctf_cs = CTFModel.from_kv(300, 2.7, defocus=-0.1)
    ctf_no_cs = CTFModel.from_kv(300, 0.0, defocus=-0.1)
    n_zeros_cs = np.sum(np.diff(np.sign(ctf_cs.simulate(freq))) != 0)
    n_zeros_no_cs = np.sum(np.diff(np.sign(ctf_no_cs.simulate(freq))) != 0)
    assert n_zeros_cs < n_zeros_no_cs


@pytest.mark.parametrize("phaseflipped", [True, False])
def test_deconvolve_preserves_contrast(phaseflipped: bool):
    ctf = CTFModel.from_kv(300, 2.7, defocus=-3.0)
    vol = _gaussian_blob((32, 64, 64))
    out = ctf.deconvolve(vol.copy(), scale=1.0, phaseflipped=phaseflipped)
    assert out.shape == vol.shape
    assert np.corrcoef(out.ravel(), vol.ravel())[0, 1] > 0


def test_deconvolve_restores_flipped_band():
    ctf = CTFModel.from_kv(300, 2.7, defocus=-3.0)
    freq = 29 / 64  # between the first and the second zeros of the CTF
    assert ctf.simulate(freq) < 0
    wave = np.cos(2 * np.pi * freq * np.arange(64))
    vol = np.tile(wave, (16, 64, 1)).astype(np.float32)
    img = ctf.apply_ctf(vol, scale=1.0)
    assert np.corrcoef(img.ravel(), vol.ravel())[0, 1] < 0
    out = ctf.deconvolve(img, scale=1.0)
    assert np.corrcoef(out.ravel(), vol.ravel())[0, 1] > 0.99
