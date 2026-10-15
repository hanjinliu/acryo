from __future__ import annotations

import numpy as np
import polars as pl
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation
import pytest

from acryo import BatchLoader, Molecules, SubtomogramLoader, TomogramSimulator
from acryo.loader import ConcatenatedLoader
from acryo.loader import _misc
from acryo.loader._group import LoaderGroup
from acryo.tilt import single_axis, dual_axis, TiltSeriesModel

SHAPES = [(9, 9, 9), (10, 10, 10), (9, 10, 11), (12, 11, 10)]


def _random_loader(
    shape, tilt_model: TiltSeriesModel | None, n: int = 7, seed: int = 0
) -> SubtomogramLoader:
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(36, 38, 40)).astype(np.float32)
    pos = rng.uniform(16, 20, size=(n, 3))
    mole = Molecules(pos, Rotation.random(n, random_state=seed))
    return SubtomogramLoader(img, mole, output_shape=shape, tilt_model=tilt_model)


def _reference(loaders: list[SubtomogramLoader], wiener_constant: float):
    """Wedge-weighted average computed with the full FFT."""
    num = 0
    den = 0
    n = 0
    for loader in loaders:
        for subvol, rot in zip(loader.asnumpy(), loader.molecules.rotator):
            mask = loader.tilt_model.create_mask(rot, subvol.shape)
            num = num + np.fft.fftn(subvol) * mask
            den = den + mask
            n += 1
    return np.fft.ifftn(num / (den + wiener_constant * n)).real


def _assert_close_to_reference(img, ref):
    """Compare the spectra, except at the Nyquist frequencies of even sizes.

    The mask at the Nyquist frequency is ambiguous (+N/2 or -N/2), so the result
    depends on how it is made Hermitian.
    """
    sel = np.ones(img.shape, dtype=bool)
    for axis, size in enumerate(img.shape):
        if size % 2 == 0:
            sl = [slice(None)] * img.ndim
            sl[axis] = size // 2
            sel[tuple(sl)] = False
    ft_img = np.fft.fftn(img)[sel]
    ft_ref = np.fft.fftn(ref)[sel]
    assert_allclose(ft_img, ft_ref, rtol=0, atol=1e-4 * np.abs(ft_ref).max())


@pytest.mark.parametrize("shape", SHAPES)
def test_no_wedge_is_mean(shape):
    loader = _random_loader(shape, tilt_model=None)
    mean = loader.average()
    out = loader.average(wedge_weighting=True, wiener_constant=0.0)
    assert out.shape == shape
    assert out.dtype == np.float32
    assert_allclose(out, mean, rtol=1e-5, atol=1e-5)
    # the Wiener constant scales the average uniformly if there is no wedge
    out = loader.average(wedge_weighting=True, wiener_constant=0.5)
    assert_allclose(out, mean / 1.5, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("wiener_constant", [0.01, 0.3])
def test_matches_reference(shape, wiener_constant):
    loader = _random_loader(shape, tilt_model=single_axis((-60, 50)))
    out = loader.average(wedge_weighting=True, wiener_constant=wiener_constant)
    ref = _reference([loader], wiener_constant)
    _assert_close_to_reference(out, ref)


def test_chunksize_independent():
    loader = _random_loader((9, 10, 11), tilt_model=single_axis((-60, 60)), n=9)
    ref = loader.average(wedge_weighting=True)
    ref_halves = loader.average_split(n_set=2, wedge_weighting=True)
    for chunksize in [64, 2, "auto"]:
        out = loader.average(wedge_weighting=True, chunksize=chunksize)
        halves = loader.average_split(
            n_set=2, wedge_weighting=True, chunksize=chunksize
        )
        assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
        assert_allclose(halves, ref_halves, rtol=1e-5, atol=1e-6)


def test_average_split():
    loader = _random_loader((9, 10, 11), tilt_model=single_axis((-60, 60)), n=10)
    halves = loader.average_split(n_set=3, n_split=2, wedge_weighting=True)
    assert halves.shape == (3, 2, 9, 10, 11)
    # each half is the weighted average of a half of the subvolumes
    rng = np.random.default_rng(0)
    for i in range(3):
        for half, sel in zip(halves[i], _misc.random_splitter(rng, 10, 2)):
            sub = loader.filter(sel)
            _assert_close_to_reference(half, _reference([sub], 0.1))


def test_negative_wiener_constant():
    loader = _random_loader((9, 9, 9), tilt_model=single_axis((-60, 60)))
    with pytest.raises(ValueError):
        loader.average(wedge_weighting=True, wiener_constant=-1.0)


def test_batch_loader_uses_each_tilt_model():
    shape = (9, 10, 11)
    loaders = [
        _random_loader(shape, single_axis((-60, 60)), n=5, seed=0),
        _random_loader(shape, dual_axis((-50, 60), (-60, 50)), n=4, seed=1),
        _random_loader(shape, None, n=3, seed=2),
    ]
    batch = BatchLoader.from_loaders(loaders, output_shape=shape)
    out = batch.average(wedge_weighting=True)
    _assert_close_to_reference(out, _reference(loaders, 0.1))
    halves = batch.average_split(wedge_weighting=True)
    assert halves.shape == (2, *shape)

    concat = ConcatenatedLoader(loaders)
    out = concat.average(wedge_weighting=True)
    _assert_close_to_reference(out, _reference(loaders, 0.1))


def test_loader_group():
    loader = _random_loader((9, 10, 11), tilt_model=single_axis((-60, 60)), n=9)
    groups = pl.Series("group", np.arange(loader.count()) % 2)
    loader = loader.replace(molecules=loader.molecules.with_features(groups))
    group = loader.groupby("group")
    assert isinstance(group, LoaderGroup)
    out = group.average(wedge_weighting=True, wiener_constant=0.2)
    for key, each in group:
        _assert_close_to_reference(out[key], _reference([each], 0.2))
    halves = group.average_split(n_set=2, wedge_weighting=True)
    for key, _ in group:
        assert halves[key].shape == (2, 2, 9, 10, 11)


def _mt_like_molecules(psi: float, centers) -> Molecules:
    """Molecules rotated around an axis in the xy plane, at angle psi from y."""
    psi = np.deg2rad(psi)
    axis = np.array([0.0, np.cos(psi), np.sin(psi)])  # zyx
    perp = np.array([0.0, -np.sin(psi), np.cos(psi)])
    phis = 2 * np.pi * np.arange(len(centers)) / 13
    z = np.stack([np.cos(p) * np.array([1.0, 0, 0]) + np.sin(p) * perp for p in phis])
    y = np.tile(axis, (len(centers), 1))
    return Molecules.from_axes(np.asarray(centers, dtype=np.float32), z=z, y=y)


def test_anisotropy_is_compensated():
    """Average of molecules rotated around an axis not parallel to the tilt axis.

    A simple average attenuates the frequencies perpendicular to the rotation axis
    more than the others, while the wedge-weighted average is close to the average
    without the missing wedge.
    """
    rng = np.random.default_rng(0)
    size = 24
    shape = (size, size, size)
    zz, yy, xx = np.indices(shape, dtype=np.float32) - (size / 2 - 0.5)
    template = np.zeros(shape, dtype=np.float32)
    for _ in range(10):
        c = rng.uniform(-5, 5, size=3)
        template += np.exp(
            -((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2) / 2.5
        )
    spacing = size + 4
    centers = [
        (size, 2 + spacing * (i + 0.5), 2 + spacing * (j + 0.5))
        for i in range(4)
        for j in range(7)
    ][:26]
    tomo_shape = (2 * size, 4 * spacing + 4, 7 * spacing + 4)
    tilt = single_axis((-60, 60))
    mole = _mt_like_molecules(60, centers)
    sim = TomogramSimulator(order=3)
    sim.add_molecules(mole, template)
    tomo = sim.simulate(tomo_shape)
    wedge = tilt.create_mask(Rotation.identity(), tomo_shape)
    tomo_wedge = np.fft.ifftn(np.fft.fftn(tomo) * wedge).real.astype(np.float32)

    ref = SubtomogramLoader(tomo, mole, output_shape=shape).average()
    loader = SubtomogramLoader(tomo_wedge, mole, output_shape=shape, tilt_model=tilt)
    plain = loader.average()
    weighted = loader.average(wedge_weighting=True, wiener_constant=0.0)

    # transfer function in the frequencies perpendicular to the rotation axis, which
    # is the y axis of the molecules
    freq = np.stack(np.meshgrid(*[np.fft.fftfreq(s) for s in shape], indexing="ij"))
    radius = np.sqrt(np.sum(freq**2, axis=0))
    shell = (radius > 0.05) & (radius < 0.3)
    equator = shell & (np.abs(freq[1]) < radius * np.sin(np.deg2rad(10)))
    axial = shell & (np.abs(freq[1]) > radius * np.cos(np.deg2rad(20)))
    ft_ref = np.fft.fftn(ref)

    def transfer(img, region):
        ft = np.fft.fftn(img)
        num = np.sum((ft * ft_ref.conj()).real[region])
        return num / np.sum(np.abs(ft_ref[region]) ** 2)

    # equatorial coverage is 2 / pi * arctan(sqrt(3) * cos(60 deg)) ~ 0.45
    assert transfer(plain, equator) < 0.6
    assert transfer(plain, axial) > 0.9
    assert transfer(weighted, equator) == pytest.approx(1.0, abs=0.1)
    assert transfer(weighted, axial) == pytest.approx(1.0, abs=0.1)
    assert np.corrcoef(plain.ravel(), ref.ravel())[0, 1] < 0.96
    assert np.corrcoef(weighted.ravel(), ref.ravel())[0, 1] > 0.98


@pytest.mark.parametrize("wiener_constant", [0.05, 0.5])
def test_fsc(wiener_constant):
    loader = _random_loader((9, 10, 11), tilt_model=single_axis((-60, 60)), n=10)
    kwargs = dict(wedge_weighting=True, wiener_constant=wiener_constant)
    halves = loader.average_split(n_set=2, seed=3, squeeze=False, **kwargs)
    df, (img0, img1), _ = loader.fsc_with_halfmaps(
        seed=3, n_set=2, zero_norm=False, **kwargs
    )
    assert_allclose(img0, halves[:, 0], rtol=1e-5, atol=1e-6)
    assert_allclose(img1, halves[:, 1], rtol=1e-5, atol=1e-6)
    assert df.columns == ["freq", "FSC-0", "FSC-1"]

    # the half maps are differently weighted from the simple ones
    _, (plain0, _), _ = loader.fsc_with_halfmaps(seed=3, n_set=2, zero_norm=False)
    assert not np.allclose(img0, plain0, atol=1e-3)

    df_avg, avg = loader.fsc_with_average(seed=3, zero_norm=False, **kwargs)
    assert_allclose(avg, halves[0].mean(axis=0), rtol=1e-5, atol=1e-6)
    df = loader.fsc(seed=3, dfreq=0.1, **kwargs)
    assert df.columns == ["freq", "FSC-0"]


def test_loader_group_fsc():
    loader = _random_loader((9, 10, 11), tilt_model=single_axis((-60, 60)), n=12)
    groups = pl.Series("group", np.arange(loader.count()) % 2)
    loader = loader.replace(molecules=loader.molecules.with_features(groups))
    group = loader.groupby("group")
    out = group.fsc(n_set=2, dfreq=0.1, wedge_weighting=True, wiener_constant=0.2)
    for key, _ in group:
        assert out[key].columns == ["freq", "FSC-0", "FSC-1"]
