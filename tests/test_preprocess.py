import dataclasses

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import ndimage as ndi

from acryo.ctf import CTFCorrection, CTFModel
from acryo.projection import TiltImagePreprocessor, fourier_bin
from acryo.projection._preprocess import (
    _axis_layout,
    _bandpass,
    _is_5_smooth,
    _reflect_indices,
    _reflect_runs,
    _subtract_plane,
)


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


def _taper(size: int, pad: int) -> np.ndarray:
    """1 on the image, and cosine falloff over the padding of the width ``pad``."""
    dist = np.maximum(np.maximum(pad - np.arange(size + 2 * pad), 0), 0)
    dist = np.maximum(dist, np.arange(size + 2 * pad) - (size + pad - 1))
    return np.cos(np.pi / 2 * np.minimum(dist / max(pad, 1), 1.0))


def _even_fft_size(size: int) -> int:
    """The smallest even size >= size that is 2 * (5-smooth number)."""
    m = -(-size // 2)
    while not _is_5_smooth(m):
        m += 1
    return 2 * m


def _reference_preprocess(image: np.ndarray, pre: TiltImagePreprocessor) -> np.ndarray:
    """Preprocessing of the whole image written step by step in numpy (float64)."""
    image = image.astype(np.float64)
    h, w = image.shape
    if pre.subtract_background:
        y, x = np.indices(image.shape)
        design = np.stack([y.ravel(), x.ravel(), np.ones(h * w)], axis=1)
        coef = np.linalg.lstsq(design, image.ravel(), rcond=None)[0]
        image = image - (design @ coef).reshape(h, w)
    pad_y, pad_x = (max(0, min(pre.padding, size - 1)) for size in image.shape)
    padded = np.pad(image, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    padded *= np.outer(_taper(h, pad_y), _taper(w, pad_x))
    # zero-padded to the size for FFT
    shape = tuple(_even_fft_size(size) for size in padded.shape)
    band = _bandpass(shape, pre.low, pre.high, pre.falloff).copy()
    if pre.low <= 0:
        band[0, 0] = 1.0  # the mean is subtracted later
    ft = np.fft.rfft2(padded, s=shape) * band
    out = np.fft.irfft2(ft, s=shape)[pad_y : pad_y + h, pad_x : pad_x + w]
    if pre.normalize:
        crop = out[int(0.375 * h) : int(0.625 * h), int(0.375 * w) : int(0.625 * w)]
        out = (out - crop.mean()) / crop.std()
    else:
        out = out - out.mean()
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
    preprocessor = TiltImagePreprocessor(patch_grid=(1, 1), **kwargs)
    ref = _reference_preprocess(image, preprocessor)
    assert_allclose(preprocessor(image), ref, atol=1e-4)


def _smooth_image(shape, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gradient = np.linspace(0, 3, shape[1])[np.newaxis, :]
    return (ndi.gaussian_filter(rng.normal(size=shape), 1.5) + gradient).astype(
        np.float32
    )


@pytest.mark.parametrize("patch_grid", [(3, 3), (2, 4), (4, 1)])
@pytest.mark.parametrize("binsize", [1, 2, 3])
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"low": 0.02, "normalize": False},
        {"overlap": 0, "subtract_background": False},
    ],
)
def test_patches_same_as_whole_image(patch_grid, binsize, kwargs):
    image = _smooth_image((260, 300))
    # band-limited below the Nyquist frequency of the binned image, otherwise the
    # Fourier binning cuts the frequencies sharply on the grids of different sizes
    high = min(0.3, 0.5 / binsize - 0.05)
    whole = TiltImagePreprocessor(
        patch_grid=(1, 1), padding=32, high=high, falloff=0.04, **kwargs
    )
    patched = whole.replace(patch_grid=patch_grid)
    ref = whole(image, binsize=binsize)
    out = patched(image, binsize=binsize)
    assert out.shape == ref.shape
    assert out.dtype == np.float32
    assert_allclose(out, ref, atol=0.02 * ref.std())


@pytest.mark.parametrize("shape", [(260, 300), (257, 301)])
@pytest.mark.parametrize("binsize", [2, 3])
def test_binning_same_as_fourier_bin(shape, binsize):
    # band-limited below the binned Nyquist frequency
    image = _smooth_image(shape)
    preprocessor = TiltImagePreprocessor(high=0.1, falloff=0.05, normalize=False)
    full = preprocessor(image)
    out = preprocessor(image, binsize=binsize)
    assert out.shape == fourier_bin(full, binsize).shape
    # Binning of band-limited images is sampling of every `binsize` pixels from the
    # same start as `fourier_bin` (but without its edge artifacts).
    slices = []
    for size, nout in zip(shape, out.shape):
        cropped = size - size % (2 * binsize)
        start = size // 2 - cropped // 2
        slices.append(slice(start, start + nout * binsize, binsize))
    ref = full[tuple(slices)]
    assert_allclose(out, ref - ref.mean(), atol=0.005 * ref.std())


@pytest.mark.parametrize("size", [5, 64, 255, 1000])
@pytest.mark.parametrize("binsize", [1, 2, 3])
@pytest.mark.parametrize("npatches", [1, 3, 4])
@pytest.mark.parametrize("overlap, padding", [(64, 128), (0, 0), (10, 7)])
def test_axis_layout(size, binsize, npatches, overlap, padding):
    if size < 2 * binsize:
        with pytest.raises(ValueError):
            _axis_layout(size, binsize, npatches, overlap, padding)
        return
    layout = _axis_layout(size, binsize, npatches, overlap, padding)
    cropped = size if binsize == 1 else size - size % (2 * binsize)
    start = size // 2 - cropped // 2
    assert layout.size_out == cropped // binsize
    assert layout.fft_size % (2 * binsize) == 0
    total = np.zeros(layout.size_out)
    for patch in layout.patches:
        assert patch.length <= layout.fft_size
        # the output pixels are on the grid of the binned patch
        assert patch.start + patch.offset * binsize == start + patch.out_start * binsize
        total[patch.out_start : patch.out_stop] += patch.weight
        n0, n1 = patch.blend
        assert np.all(patch.weight[n0 : patch.weight.size - n1] == 1)
        assert patch.out_start * binsize <= patch.center - start
        assert patch.center - start <= patch.out_stop * binsize
    # the weights are a partition of unity
    assert_allclose(total, 1.0, rtol=1e-6)
    assert layout.patches[0].out_start == 0
    assert layout.patches[-1].out_stop == layout.size_out


@pytest.mark.parametrize("size", [1, 2, 7, 30])
def test_reflect_indices_and_runs(size):
    image = np.arange(size * 3, dtype=np.float32).reshape(size, 3)
    padded = np.pad(np.arange(size), 3 * size, mode="reflect")
    for start in range(-2 * size, 2 * size):
        for length in [1, size, 2 * size]:
            idx = _reflect_indices(start, length, size)
            assert_allclose(idx, padded[start + 3 * size : start + 3 * size + length])
            runs = _reflect_runs(start, length, size)
            if runs is None:
                continue
            out = np.full((length, 3), np.nan, dtype=np.float32)
            for dst, src in runs:
                out[dst] = image[src]
            assert_allclose(out, image[idx])


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


@pytest.mark.parametrize("patch_grid", [(1, 1), (3, 3)])
def test_filters_are_multiplied(patch_grid):
    image = _image((50, 60))
    base = TiltImagePreprocessor(normalize=False, padding=8, patch_grid=patch_grid)
    calls = []

    def _double(freq_y, freq_x, index):
        calls.append((freq_y.shape, freq_x.shape, index))
        return 2.0

    out = base.with_filters(_double)(image, index=3, scale=0.5)
    assert_allclose(out, base(image) * 2, rtol=1e-5, atol=1e-5)
    # called once for each image
    if patch_grid == (1, 1):
        # padded to (66, 76), and to the sizes for FFT
        assert calls == [((72, 1), (1, 41), 3)]
    else:
        assert len(calls) == 1


def test_filter_frequency_unit():
    seen = {}

    def _record(freq_y, freq_x, index):
        seen["max"] = (np.abs(freq_y).max(), np.abs(freq_x).max())
        return 1.0

    preprocessor = TiltImagePreprocessor(padding=0).with_filters(_record)
    preprocessor(_image((40, 40)), scale=0.25)
    assert_allclose(seen["max"], [0.5 / 0.25, 0.5 / 0.25])  # Nyquist in 1/nm
    # only the frequencies kept by binning
    preprocessor(_image((40, 40)), scale=0.25, binsize=2)
    assert_allclose(seen["max"], [0.5 / 0.5, 0.5 / 0.5])


class _PositionRecorder:
    """A position-dependent filter that scales each patch by a factor."""

    position_dependent = True

    def __init__(self, factor=lambda position: 1.0):
        self.calls: list[tuple[int, tuple[float, float]]] = []
        self.factor = factor

    def __call__(self, freq_y, freq_x, index, position):
        self.calls.append((index, position))
        return self.factor(position)


@pytest.mark.parametrize("binsize", [1, 2])
def test_position_dependent_filter(binsize):
    shape, scale = (90, 120), 0.5
    recorder = _PositionRecorder()
    preprocessor = TiltImagePreprocessor(patch_grid=(3, 4)).with_filters(recorder)
    preprocessor(_image(shape), index=2, scale=scale, binsize=binsize)
    assert [index for index, _ in recorder.calls] == [2] * 12
    positions = np.array([position for _, position in recorder.calls])
    # centers of the patches in nm from the center of the image
    for axis, (size, npatches) in enumerate(zip(shape, (3, 4))):
        coords = np.unique(np.round(positions[:, axis], 6))
        expected = (np.arange(npatches) + 0.5) * size / npatches - size // 2
        assert_allclose(coords, expected * scale, atol=binsize * scale)


def test_position_dependent_filter_is_applied_to_each_patch():
    image = _smooth_image((120, 150))
    # (y, x) factors of each patch of the 2x3 grid
    factors = {(0, 0): 1.0, (0, 1): 2.0, (0, 2): 3.0, (1, 0): -1.0, (1, 1): 0.5}

    def _factor(position):
        key = (int(position[0] > 0), int(np.round(position[1] / 50 + 1)))
        return factors.get(key, 4.0)

    plain = TiltImagePreprocessor(patch_grid=(2, 3), overlap=10, normalize=False)
    scaled = plain.with_filters(_PositionRecorder(_factor))
    out_plain, out_scaled = plain(image), scaled(image)
    # the cores of the patches (without the overlaps) are scaled up to a constant
    for iy, ys in enumerate([slice(0, 55), slice(65, 120)]):
        for ix, xs in enumerate([slice(0, 45), slice(55, 95), slice(105, 150)]):
            factor = factors.get((iy, ix), 4.0)
            diff = out_scaled[ys, xs] - factor * out_plain[ys, xs]
            assert diff.std() < 1e-4 * out_plain.std()


@pytest.mark.parametrize("shape", [(2, 3), (5, 7), (17, 4)])
@pytest.mark.parametrize("patch_grid", [(1, 1), (3, 3), (8, 8)])
def test_small_images(shape, patch_grid):
    image = _image(shape)
    preprocessor = TiltImagePreprocessor(
        patch_grid=patch_grid, normalize=False, high=1.0
    )
    out = preprocessor(image)
    assert out.shape == shape
    assert np.all(np.isfinite(out))
    # the bandpass filter passes all the frequencies
    ref = _subtract_plane(image)
    assert_allclose(out, ref - ref.mean(), atol=1e-4)


CTF = CTFModel.from_kv(300, spherical_aberration=2.7, defocus=-2.0)


def test_ctf_correction_values():
    freq = np.linspace(0, 2.0, 100)
    flip = CTFCorrection(CTF)
    assert flip.models == (CTF,)
    ctf = CTF.simulate(freq)
    # computed in float32
    assert flip(freq, 0.0, index=5).dtype == np.float32
    assert_allclose(flip(freq, 0.0, index=5), np.sign(ctf))
    wiener = CTFCorrection(CTF, method="wiener", wiener_constant=0.2)
    assert_allclose(wiener(0.0, freq, index=0), ctf / (ctf**2 + 0.2), atol=1e-4)


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
    with pytest.raises(ValueError):
        TiltImagePreprocessor(overlap=-1)
    with pytest.raises(ValueError):
        TiltImagePreprocessor(patch_grid=0)
    with pytest.raises(ValueError):
        TiltImagePreprocessor(patch_grid=(1, 2, 3))  # type: ignore
    with pytest.raises(ValueError):
        TiltImagePreprocessor()(_image((10, 10)), binsize=0)
    with pytest.raises(ValueError):
        TiltImagePreprocessor()(_image((10, 10)), binsize=1.5)  # type: ignore
    with pytest.raises(ValueError):
        TiltImagePreprocessor()(_image((10, 10)), binsize=6)  # too small to bin
    assert TiltImagePreprocessor(patch_grid=2).patch_grid == (2, 2)


def test_real_space_binsize():
    from acryo.projection._preprocess import _real_space_binsize

    k3 = (4092, 5760)  # gcd is 12
    assert [_real_space_binsize(k3, b, 4) for b in (2, 3, 4, 6, 7, 8)] == [
        1,
        1,
        4,
        6,
        6,
        6,
    ]
    assert _real_space_binsize((4096, 4096), 8, 4) == 8
    assert _real_space_binsize(k3, 8, None) == 1
    assert _real_space_binsize((277, 373), 8, 4) == 1  # no common factor
    with pytest.raises(ValueError):
        TiltImagePreprocessor(real_space_binning=1)


def test_bin_real_space():
    from acryo.projection._preprocess import _bin_real_space

    image = _image((24, 36))
    expected = image.reshape(4, 6, 6, 6).mean(axis=(1, 3))
    assert_allclose(_bin_real_space(image, 6), expected, rtol=1e-5)


@pytest.mark.parametrize("size", [276, 372, 4092])
@pytest.mark.parametrize("binsize, prebin", [(4, 4), (5, 4), (7, 6), (8, 6), (9, 6)])
@pytest.mark.parametrize("npatches", [1, 3])
def test_axis_layout_real_space_binning(size, binsize, prebin, npatches):
    from fractions import Fraction

    layout = _axis_layout(size, binsize, npatches, 64, 128, prebin=prebin)
    cropped = size - size % (2 * binsize)
    assert layout.size_out == cropped // binsize
    step = Fraction(binsize, prebin)
    # the FFT of the size p * m is cropped to the even size q * m
    assert layout.fft_size * step.denominator == layout.crop_size * step.numerator
    assert layout.crop_size % 2 == 0
    total = np.zeros(layout.size_out)
    for patch in layout.patches:
        assert patch.length <= layout.fft_size
        assert 0 <= patch.shift < 1
        nout = patch.out_stop - patch.out_start
        assert patch.offset * 2 + nout <= layout.crop_size
        # the samples of the binned patch are at the output pixels
        for k in [0, nout - 1]:
            j = patch.out_start + k
            sample = patch.start + patch.shift + float((patch.offset + k) * step)
            original = size // 2 + (j - layout.size_out // 2) * binsize
            assert sample * prebin + (prebin - 1) / 2 == pytest.approx(original)
        assert patch.taper.shape == (patch.length,)
        assert np.all(
            patch.taper[patch.margins[0] : patch.length - patch.margins[1]] == 1
        )
        total[patch.out_start : patch.out_stop] += patch.weight
    assert_allclose(total, 1.0, rtol=1e-6)


def _blob(shape, shift, sigma) -> np.ndarray:
    pos = np.array(shape) // 2 + shift
    yy, xx = np.indices(shape, dtype=np.float32)
    out = np.exp(-((yy - pos[0]) ** 2 + (xx - pos[1]) ** 2) / (2 * sigma**2))
    return out.astype(np.float32)


@pytest.mark.parametrize("shape", [(276, 372), (240, 336)])
@pytest.mark.parametrize("binsize", [4, 5, 7, 8, 9])
@pytest.mark.parametrize("patch_grid", [(1, 1), (3, 3)])
def test_real_space_binning_keeps_center(shape, binsize, patch_grid):
    shift = np.array([13.0, -21.4])
    image = _blob(shape, shift, 10.0)
    preprocessor = TiltImagePreprocessor(
        high=1.0, subtract_background=False, normalize=False, patch_grid=patch_grid
    )
    out = preprocessor(image, binsize=binsize)
    fourier = preprocessor.replace(real_space_binning=None)(image, binsize=binsize)
    assert out.shape == fourier.shape
    weight = np.clip(out - out.min(), 0, None) ** 2
    com = (np.indices(out.shape) * weight).sum(axis=(1, 2)) / weight.sum()
    assert_allclose(com, np.array(out.shape) // 2 + shift / binsize, atol=0.02)
    # only slightly different from the binning in Fourier space
    assert np.abs(out - fourier).max() < 0.05 * np.abs(fourier).max()


@pytest.mark.parametrize("binsize", [4, 8])
def test_real_space_binning_low_frequency(binsize):
    # band-limited well below the Nyquist frequency of the binned image, where the
    # real space binning does not attenuate or alias much
    rng = np.random.default_rng(3)
    image = ndi.gaussian_filter(rng.normal(size=(264, 360)), 1.5 * binsize)
    image = image.astype(np.float32)
    preprocessor = TiltImagePreprocessor()
    out = preprocessor(image, binsize=binsize)
    fourier = preprocessor.replace(real_space_binning=None)(image, binsize=binsize)
    assert np.corrcoef(out.ravel(), fourier.ravel())[0, 1] > 0.995
