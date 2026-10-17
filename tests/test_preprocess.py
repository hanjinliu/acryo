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
def test_same_as_torch(shape, kwargs):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_tilt_series")
    from torch_tilt_series import preprocess_tilt_series_images

    image = _image(shape)
    preprocessor = TiltImagePreprocessor(**kwargs)
    torch_kwargs = {
        ("bandpass_padding" if k == "padding" else k): v for k, v in kwargs.items()
    }
    if preprocessor.subtract_background:
        # the plane fitting of torch is less accurate in float32
        torch_kwargs["subtract_background"] = False
        torch_input = _subtract_plane(image)
    else:
        torch_input = image
    ref = preprocess_tilt_series_images(torch.as_tensor(torch_input), **torch_kwargs)
    assert_allclose(preprocessor(image), ref.numpy(), atol=1e-5)


@pytest.mark.parametrize("shape", [(30, 40), (31, 43)])
@pytest.mark.parametrize("pad", [0, 5])
def test_soft_rectangle_same_as_torch(shape, pad):
    pytest.importorskip("torch")
    from torch_grid_utils.shapes_2d import rectangle

    padded = (shape[0] + 2 * pad, shape[1] + 2 * pad)
    ref = rectangle(dimensions=shape, image_shape=padded, smoothing_radius=pad)
    assert_allclose(_soft_rectangle(shape, pad), ref.numpy(), atol=1e-6)


@pytest.mark.parametrize("shape", [(30, 40), (31, 43)])
@pytest.mark.parametrize(
    "low, high, falloff", [(0.0, 0.5, 0.0), (0.05, 0.3, 0.0), (0.05, 0.3, 0.04)]
)
def test_bandpass_same_as_torch(shape, low, high, falloff):
    pytest.importorskip("torch")
    from torch_fourier_filter.bandpass import bandpass_filter

    ref = bandpass_filter(low, high, falloff, shape, rfft=True, fftshift=False)
    assert_allclose(_bandpass(shape, low, high, falloff), ref.numpy(), atol=1e-5)


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
def test_fourier_bin_same_as_torch(shape, binsize):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_fourier_rescale")
    from torch_fourier_rescale import fourier_rescale_2d

    image = _image(shape)
    slices = []
    for size in shape:
        cropped = size - size % (2 * binsize)
        start = size // 2 - cropped // 2
        slices.append(slice(start, start + cropped))
    cropped = image[tuple(slices)]
    target = tuple(s // binsize for s in cropped.shape)
    ref, _ = fourier_rescale_2d(torch.as_tensor(cropped), target_shape=target)
    assert_allclose(fourier_bin(image, binsize), ref.numpy(), atol=1e-5)
    assert fourier_bin(image, 1) is not None
    with pytest.raises(ValueError):
        fourier_bin(image, 0)


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
