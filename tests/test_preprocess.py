import dataclasses

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import ndimage as ndi

from acryo.ctf import CTFModel
from acryo.projection import CTFCorrection, TiltImagePreprocessor, fourier_bin
from acryo.projection._preprocess import _bandpass, _soft_rectangle, _subtract_plane


def _image(shape, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gradient = np.linspace(0, 5, shape[1])[np.newaxis, :]
    return (rng.normal(size=shape) + gradient).astype(np.float32)


def test_subtract_plane():
    image = _image((60, 70))
    h, w = image.shape
    y, x = np.indices(image.shape)
    design = np.stack([y.ravel(), x.ravel(), np.ones(h * w)], axis=1)
    coef = np.linalg.lstsq(design, image.ravel().astype(np.float64), rcond=None)[0]
    expected = image - (design @ coef).reshape(h, w)
    assert_allclose(_subtract_plane(image), expected, atol=1e-5)


def _reference_preprocess(image: np.ndarray, pre: TiltImagePreprocessor) -> np.ndarray:
    """Preprocessing written step by step in numpy (float64)."""
    image = image.astype(np.float64)
    h, w = image.shape
    if pre.subtract_background:
        y, x = np.indices(image.shape)
        design = np.stack([y.ravel(), x.ravel(), np.ones(h * w)], axis=1)
        coef = np.linalg.lstsq(design, image.ravel(), rcond=None)[0]
        image = image - (design @ coef).reshape(h, w)
    pad = max(0, min(pre.padding, h - 1, w - 1))
    padded = np.pad(image, pad, mode="reflect") * _soft_rectangle((h, w), pad)
    ft = np.fft.rfft2(padded) * _bandpass(padded.shape, pre.low, pre.high, pre.falloff)
    out = np.fft.irfft2(ft, s=padded.shape)[pad : pad + h, pad : pad + w]
    if pre.normalize:
        crop = out[int(0.375 * h) : int(0.625 * h), int(0.375 * w) : int(0.625 * w)]
        out = (out - crop.mean()) / crop.std()
    return out


@pytest.mark.parametrize("shape", [(60, 70), (61, 73)])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"low": 0.02, "high": 0.4, "falloff": 0.05},
        {"subtract_background": False, "normalize": False},
        {"padding": 0},
        {"padding": 8},
    ],
)
def test_preprocess(shape, kwargs):
    image = _image(shape)
    preprocessor = TiltImagePreprocessor(**kwargs)
    ref = _reference_preprocess(image, preprocessor)
    assert_allclose(preprocessor(image), ref, atol=1e-4)


@pytest.mark.parametrize("shape", [(30, 40), (31, 43)])
@pytest.mark.parametrize("pad", [0, 5])
def test_soft_rectangle(shape, pad):
    mask = _soft_rectangle(shape, pad)
    assert mask.shape == (shape[0] + 2 * pad, shape[1] + 2 * pad)
    # symmetric around the center (size // 2) of the padded image
    cy, cx = np.array(mask.shape) // 2
    ny = min(cy, mask.shape[0] - 1 - cy)
    nx = min(cx, mask.shape[1] - 1 - cx)
    sub = mask[cy - ny : cy + ny + 1, cx - nx : cx + nx + 1]
    assert_allclose(sub, sub[::-1, ::-1])
    # 1 strictly within half the size from the center, cosine edge of the width pad
    for axis, size in enumerate(shape):
        line = mask[:, cx] if axis == 0 else mask[cy, :]
        coords = np.arange(line.size) - line.size // 2
        inside = np.abs(coords) < size / 2
        assert_allclose(line[inside], 1.0)
        dist = np.abs(coords) - np.abs(coords[inside]).max()
        edge = ~inside & (dist <= pad)
        assert_allclose(line[edge], np.cos(np.pi / 2 * dist[edge] / pad), atol=1e-6)
        assert_allclose(line[dist > pad], 0.0)
    if shape[0] % 2 == 1 and shape[1] % 2 == 1:
        # odd-sized images are exactly the region of the value 1
        assert_allclose(mask[pad : pad + shape[0], pad : pad + shape[1]], 1.0)


@pytest.mark.parametrize("shape", [(30, 40), (31, 43)])
@pytest.mark.parametrize(
    "low, high, falloff", [(0.0, 0.5, 0.0), (0.05, 0.3, 0.0), (0.05, 0.3, 0.04)]
)
def test_bandpass(shape, low, high, falloff):
    out = _bandpass(shape, low, high, falloff)
    assert out.shape == (shape[0], shape[1] // 2 + 1)
    freq = np.hypot(
        np.fft.fftfreq(shape[0])[:, np.newaxis], np.fft.rfftfreq(shape[1])[np.newaxis]
    )
    # frequencies at the edges of the band are ambiguous in float32
    edge = np.isclose(freq, low, atol=1e-6) | np.isclose(freq, high, atol=1e-6)
    band = (freq > low) & (freq <= high) & ~edge
    out_band = (~band) & ~edge
    assert out[0, 0] == 0.0  # the mean is removed
    assert_allclose(out[band], 1.0)
    if falloff == 0:
        assert_allclose(out[out_band], 0.0)
    else:
        dist = np.maximum(low - freq, freq - high)
        soft = out_band & (dist < falloff)
        assert_allclose(out[soft], np.cos(np.pi / 2 * dist[soft] / falloff), atol=1e-5)
        assert_allclose(out[out_band & (dist > falloff)], 0.0)


@pytest.mark.parametrize("shape", [(64, 80), (63, 81), (66, 70)])
@pytest.mark.parametrize("binsize", [2, 3])
def test_fourier_bin_keeps_center(shape, binsize):
    shift = np.array([6.0, -9.0])  # from the center, in the original pixels
    pos = np.array(shape) // 2 + shift
    yy, xx = np.indices(shape, dtype=np.float32)
    image = np.exp(-((yy - pos[0]) ** 2 + (xx - pos[1]) ** 2) / (2 * 3.0**2))
    binned = fourier_bin(image, binsize)
    assert binned.dtype == np.float32
    expected = np.array(binned.shape) // 2 + shift / binsize
    weight = np.clip(binned, 0, None) ** 2
    com = (np.indices(binned.shape) * weight).sum(axis=(1, 2)) / weight.sum()
    assert_allclose(com, expected, atol=0.05)
    # the mean is preserved
    assert binned.mean() == pytest.approx(
        image[: binned.shape[0] * binsize, : binned.shape[1] * binsize].mean(), rel=0.05
    )


@pytest.mark.parametrize("shape", [(64, 80), (63, 81)])
@pytest.mark.parametrize("binsize", [2, 3])
def test_fourier_bin_band_limited(shape, binsize):
    # periodic in the cropped image and band-limited below the binned Nyquist
    cropped = [size - size % (2 * binsize) for size in shape]
    target = [c // binsize for c in cropped]

    def func(y, x):
        out = np.cos(2 * np.pi * (2 * y / cropped[0] + 0.3))
        out += 0.5 * np.sin(2 * np.pi * (3 * x / cropped[1] - y / cropped[0]))
        return out

    # coordinates relative to the center (size // 2), which is kept
    yy, xx = np.indices(shape) - (np.array(shape) // 2)[:, None, None]
    binned = fourier_bin(func(yy, xx).astype(np.float32), binsize)
    assert binned.shape == tuple(target)
    yb, xb = (np.indices(target) - (np.array(target) // 2)[:, None, None]) * binsize
    assert_allclose(binned, func(yb, xb), atol=1e-4)
    assert fourier_bin(binned, 1) is binned
    with pytest.raises(ValueError):
        fourier_bin(binned, 0)


def test_filters_are_multiplied():
    image = _image((50, 60))
    base = TiltImagePreprocessor(normalize=False, padding=8)
    calls = []

    def _double(freq_y, freq_x, index):
        calls.append((freq_y.shape, freq_x.shape, index))
        return 2.0

    out = base.with_filters(_double)(image, index=3, scale=0.5)
    assert_allclose(out, base(image) * 2, rtol=1e-5, atol=1e-5)
    assert calls == [((66, 1), (1, 39), 3)]


def test_filter_frequency_unit():
    seen = {}

    def _record(freq_y, freq_x, index):
        seen["max"] = (np.abs(freq_y).max(), np.abs(freq_x).max())
        return 1.0

    TiltImagePreprocessor(padding=0).with_filters(_record)(_image((40, 40)), scale=0.25)
    assert_allclose(seen["max"], [0.5 / 0.25, 0.5 / 0.25])  # Nyquist in 1/nm


CTF = CTFModel.from_kv(300, spherical_aberration=2.7, defocus=-2.0)


def test_ctf_correction_values():
    freq = np.linspace(0, 2.0, 100)
    flip = CTFCorrection(CTF)
    assert flip.models == (CTF,)
    ctf = CTF.simulate(freq)
    assert_allclose(flip(freq, 0.0, index=5), np.sign(ctf))
    wiener = CTFCorrection(CTF, method="wiener", wiener_constant=0.2)
    assert_allclose(wiener(0.0, freq, index=0), ctf / (ctf**2 + 0.2))


def test_ctf_correction_per_tilt():
    defocus = [-1.0, -2.0, -3.0]
    correction = CTFCorrection.from_defocus(CTF, defocus)
    assert [m.defocus for m in correction.models] == defocus
    freq = np.linspace(0, 2.0, 100)
    for index, d in enumerate(defocus):
        expected = np.sign(dataclasses.replace(CTF, defocus=d).simulate(freq))
        assert_allclose(correction(freq, 0.0, index), expected)
    with pytest.raises(IndexError):
        correction(freq, 0.0, 3)


def test_select_tilts():
    defocus = [-1.0, -2.0, -3.0, -4.0]
    calls: list[int] = []

    def _filter(freq_y, freq_x, index):
        calls.append(index)
        return 1.0

    preprocessor = (
        TiltImagePreprocessor(low=0.01)
        .with_ctf(CTF, defocus=defocus)
        .with_ctf(CTF)
        .with_filters(_filter)
    )
    selected = preprocessor.select_tilts([3, 1])
    assert selected.low == 0.01
    per_tilt, single, generic = selected.filters
    # the CTF models of the selected tilts
    assert [m.defocus for m in per_tilt.models] == [-4.0, -2.0]
    assert single is preprocessor.filters[1]
    # other filters are called with the indices before the selection
    generic(0.0, 0.0, 0)
    generic(0.0, 0.0, 1)
    generic.select_tilts([1])(0.0, 0.0, 0)
    assert calls == [3, 1, 1]
    image = _image((32, 40))
    assert_allclose(selected(image, index=0), preprocessor(image, index=3))


def test_ctf_correction_invalid_input():
    with pytest.raises(ValueError):
        CTFCorrection(())
    with pytest.raises(ValueError):
        CTFCorrection(CTF, method="invalid")  # type: ignore
    with pytest.raises(ValueError):
        CTFCorrection(CTF, method="wiener", wiener_constant=0.0)


def test_with_ctf():
    preprocessor = TiltImagePreprocessor(low=0.01)
    with_ctf = preprocessor.with_ctf(CTF, defocus=[-1.0, -2.0], method="wiener")
    assert preprocessor.filters == ()
    (correction,) = with_ctf.filters
    assert isinstance(correction, CTFCorrection)
    assert correction.method == "wiener"
    assert [m.defocus for m in correction.models] == [-1.0, -2.0]
    assert with_ctf.low == 0.01
    models = [CTF, dataclasses.replace(CTF, defocus=-3.0)]
    assert preprocessor.with_ctf(models).filters[0].models == tuple(models)


@pytest.mark.parametrize("method", ["phase_flip", "wiener"])
def test_ctf_correction_restores_contrast(method):
    scale = 0.2  # nm
    rng = np.random.default_rng(8)
    shape = (128, 128)
    points = np.zeros(shape, dtype=np.float32)
    points[tuple(rng.integers(10, 118, size=(2, 30)))] = 1.0
    image = ndi.gaussian_filter(points, 1.5)
    observed = CTF.apply_ctf(image, scale).astype(np.float32)
    plain = TiltImagePreprocessor(subtract_background=False, padding=16)
    corrected = plain.with_ctf(CTF, method=method)

    def _corr(img):
        return np.corrcoef(img.ravel(), image.ravel())[0, 1]

    assert _corr(corrected(observed, scale=scale)) > _corr(plain(observed, scale=scale))
    assert _corr(corrected(observed, scale=scale)) > 0.5


def test_invalid_preprocessor():
    with pytest.raises(ValueError):
        TiltImagePreprocessor(padding=-1)
    with pytest.raises(ValueError):
        TiltImagePreprocessor(falloff=-0.1)
    with pytest.raises(ValueError):
        TiltImagePreprocessor()(np.zeros((3, 4, 5)))


@pytest.mark.parametrize("shape", [(2, 3), (5, 7), (64, 81), (100, 37)])
@pytest.mark.parametrize("subtract_background", [False, True])
def test_pad_reflect(shape, subtract_background):
    from acryo.projection._preprocess import _pad_reflect

    image = _image(shape)
    for pad in sorted({0, 1, min(shape) - 1, min(4, min(shape) - 1)}):
        ref = _subtract_plane(image) if subtract_background else image
        ref = np.pad(ref, pad, mode="reflect")
        out = _pad_reflect(image, pad, subtract_background)
        assert out.dtype == np.float32
        assert_allclose(out, ref, atol=1e-6)
