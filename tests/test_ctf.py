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
