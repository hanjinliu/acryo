import numpy as np
from numpy.testing import assert_allclose
import polars as pl
import pytest
from scipy.spatial.transform import Rotation

from acryo import Molecules, SubtomogramLoader, BackProjectionLoader, BatchLoader
from acryo.backend import Backend
from acryo.projection import (
    LocalAlignment,
    LocalShifts2D,
    TiltImagePreprocessor,
    TiltSeries,
)
from acryo.tilt import SingleAxis, single_axis

mrcfile = pytest.importorskip("mrcfile")

IMAGE_SCALE = 0.5  # pixel size of the tilt images in nm
IMAGE_SHAPE = (128, 144)
TILT_ANGLES = np.arange(-40.0, 61.0, 4.0)


@pytest.fixture(autouse=True)
def skip_if_cupy():
    if Backend().name == "cupy":
        pytest.skip(reason="not tested for cupy yet")


def _make_tilt_series(
    path, points_nm: np.ndarray, sigma: float = 1.5, local_shifts_2d=None
) -> TiltSeries:
    """Simulate a tilt series of Gaussian blobs at the given points.

    The points are (z, y, x) coordinates in nm relative to the tomogram center. As
    in cryo-EM images, the blobs are darker than the background.
    """
    rng = np.random.default_rng(0)
    ntilts = TILT_ANGLES.size
    ts = TiltSeries(
        tilt_angles=TILT_ANGLES,
        tilt_axis_angle=85.0,
        sample_translations=rng.normal(scale=1.0, size=(ntilts, 2)),
        scale=IMAGE_SCALE,
        image_path=path,
        local_shifts_2d=local_shifts_2d,
    )
    projected = ts.project_points(points_nm) / IMAGE_SCALE
    projected = projected + np.array(IMAGE_SHAPE) // 2
    yy, xx = np.indices(IMAGE_SHAPE, dtype=np.float32)
    images = np.zeros((ntilts,) + IMAGE_SHAPE, dtype=np.float32)
    for proj in projected:
        for t, (py, px) in enumerate(proj):
            images[t] -= np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * sigma**2))
    with mrcfile.new(path, overwrite=True) as f:
        f.set_data(images)
    return ts


CENTER = np.array([10.0, 30.0, 35.0])  # tomogram center in nm
BLOBS = np.array(
    [
        [0.0, 0.0, 0.0],
        [3.0, -15.0, 18.0],
        [-4.0, 12.0, -16.0],
        [5.0, 14.0, 14.0],
        [-2.0, -13.0, -15.0],
    ]
)  # relative to the tomogram center, in nm


@pytest.fixture
def blob_tilt_series(tmp_path) -> TiltSeries:
    return _make_tilt_series(tmp_path / "blobs.mrc", BLOBS)


def _loader(ts: TiltSeries, mole: Molecules, **kwargs) -> BackProjectionLoader:
    kwargs.setdefault("output_shape", (13, 13, 13))
    # the pixel size of the subtomograms
    scale = kwargs.pop("scale", IMAGE_SCALE)
    return BackProjectionLoader(
        ts, mole, tomogram_center=CENTER, binsize=scale / IMAGE_SCALE, **kwargs
    )


def _argmax(img: np.ndarray) -> tuple[int, ...]:
    return tuple(int(i) for i in np.unravel_index(np.argmax(img), img.shape))


def _mirror_z(points: np.ndarray) -> np.ndarray:
    """Mirror points along the z axis around the origin."""
    return points * np.array([-1.0, 1.0, 1.0])


def _mirror_rotation(rot: Rotation) -> Rotation:
    """Rotation in the coordinate system mirrored along the z axis."""
    flip = np.diag([-1.0, 1.0, 1.0])
    return Rotation.from_matrix(flip @ rot.as_matrix() @ flip)


def _center_of_mass(img: np.ndarray) -> np.ndarray:
    weight = np.clip(img.astype(np.float64), 0, None) ** 4
    inds = np.indices(img.shape).reshape(3, -1)
    return (inds * weight.ravel()).sum(axis=1) / weight.sum()


# NOTE: scales are chosen so that the pixel sizes are both even and odd multiples of
# the tilt images.
@pytest.mark.parametrize("scale", [0.5, 0.64, 0.8, 1.0])
@pytest.mark.parametrize("order", [0, 1, 3])
def test_subtomograms_are_centered(blob_tilt_series, scale, order):
    rot = Rotation.random(len(BLOBS), random_state=1)
    mole = Molecules(BLOBS + CENTER, rot)
    loader = _loader(blob_tilt_series, mole, scale=scale, order=order)
    stack = loader.asnumpy()
    assert stack.shape == (len(BLOBS), 13, 13, 13)
    assert stack.dtype == np.float32
    for subtomo in stack:
        assert _argmax(subtomo) == (6, 6, 6)
        assert_allclose(_center_of_mass(subtomo), [6, 6, 6], atol=0.1)


@pytest.mark.parametrize("corner_safe", [False, True])
def test_subtomograms_are_rotated(tmp_path, corner_safe):
    # blobs are at the local (0, 0, 2) nm position of the molecules
    rot = Rotation.random(len(BLOBS), random_state=2)
    offset = np.array([0.0, 0.0, 2.0])
    ts = _make_tilt_series(tmp_path / "rotated.mrc", BLOBS + rot.apply(offset))
    mole = Molecules(BLOBS + CENTER, rot)
    loader = _loader(ts, mole, output_shape=(11, 12, 15), corner_safe=corner_safe)
    for subtomo in loader.asnumpy():
        # center is (5, 5.5, 7)
        assert_allclose(_center_of_mass(subtomo), [5, 5.5, 11], atol=0.15)


@pytest.mark.parametrize("tomo_hand", [1, -1])
@pytest.mark.parametrize("scale", [0.5, 0.8])  # even and odd depth
def test_consistent_with_tomogram_reconstruction(blob_tilt_series, scale, tomo_hand):
    tomo_shape = tuple(int(np.ceil(2 * c / scale)) for c in CENTER)
    tomo_center = np.array(tomo_shape) // 2 * scale
    rot = Rotation.random(len(BLOBS), random_state=3)
    blobs = BLOBS if tomo_hand == 1 else _mirror_z(BLOBS)
    mole = Molecules(blobs + tomo_center, rot)
    kwargs = dict(output_shape=(14, 15, 16), scale=scale, order=1)
    loader = BackProjectionLoader(
        blob_tilt_series,
        mole,
        tomogram_center=tomo_center,
        tomo_hand=tomo_hand,
        output_shape=(14, 15, 16),
        order=1,
        binsize=scale / IMAGE_SCALE,
    )
    assert loader.scale == pytest.approx(scale)
    # NOTE: the contrast of the tomogram is not inverted
    tomo = -loader.reconstruct_tomogram(tomo_shape, patch_size=40)
    ref = SubtomogramLoader(tomo, mole, **kwargs).asnumpy()
    out = loader.asnumpy()
    for img0, img1 in zip(ref, out):
        corr = np.corrcoef(img0.ravel(), img1.ravel())[0, 1]
        assert corr > 0.92  # half-pixel shift gives ~0.87


def test_loading_methods_are_consistent(blob_tilt_series):
    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=4))
    loader = _loader(blob_tilt_series, mole)
    ref = loader.asnumpy()
    each = np.stack([arr.compute() for arr in loader.construct_loading_tasks()])
    assert_allclose(each, ref, rtol=1e-6, atol=1e-6)
    assert_allclose(loader.load([3, 1]), ref[[3, 1]], rtol=1e-6, atol=1e-6)
    assert_allclose(loader.load(2), ref[2], rtol=1e-6, atol=1e-6)
    assert_allclose(np.stack(list(loader.load_iter())), ref, rtol=1e-6, atol=1e-6)
    assert_allclose(loader.average(), ref.mean(axis=0), rtol=1e-5, atol=1e-5)


@pytest.fixture
def recorded_calls(monkeypatch) -> list[tuple[int, int]]:
    """Record the number of cubes and the sidelength of each reconstruction."""
    from acryo.projection import DirectFourierReconstructor

    calls: list[tuple[int, int]] = []
    _call = DirectFourierReconstructor.__call__

    def _call_record(self, positions):
        calls.append((len(positions), self.sidelength))
        return _call(self, positions)

    monkeypatch.setattr(DirectFourierReconstructor, "__call__", _call_record)
    return calls


@pytest.mark.parametrize(
    "batch_size, npoints", [(1, [1] * 5), (2, [2, 2, 1]), (None, [1] * 5)]
)
def test_batch_size(blob_tilt_series, recorded_calls, batch_size, npoints):
    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=4))
    ref = _loader(blob_tilt_series, mole, batch_size=len(BLOBS)).asnumpy()
    recorded_calls.clear()
    loader = _loader(blob_tilt_series, mole, batch_size=batch_size)
    stack = loader.construct_dask()
    assert recorded_calls == []  # lazy
    assert stack.chunks[0] == tuple(npoints)
    assert_allclose(stack.compute(), ref, rtol=1e-5, atol=1e-5)
    assert sorted(n for n, _ in recorded_calls) == sorted(npoints)


def test_load_reconstructs_given_subtomograms(blob_tilt_series, recorded_calls):
    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=4))
    loader = _loader(blob_tilt_series, mole, batch_size=8)
    loader.load(2)
    loader.load([3, 1])
    loader.load(slice(1, 5, 2))
    assert [n for n, _ in recorded_calls] == [1, 2, 2]


def test_reconstructor_is_cached(blob_tilt_series, monkeypatch):
    from acryo.projection import DirectFourierReconstructor

    sidelengths: list[int] = []
    _init = DirectFourierReconstructor.__init__

    def _init_record(self, *args, **kwargs):
        _init(self, *args, **kwargs)
        sidelengths.append(self.sidelength)

    monkeypatch.setattr(DirectFourierReconstructor, "__init__", _init_record)
    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=4))
    loader = _loader(blob_tilt_series, mole, order=1)
    loader.asnumpy()
    loader.load(1)
    assert sidelengths == [15, 15]  # not cached by default
    sidelengths.clear()
    with loader.cache_tilt_images():
        loader.asnumpy()
        loader.load(1)
        loader.load([0, 2])
        assert sidelengths == [15]  # 13 + 2 * order
        loader.asnumpy(output_shape=(9, 9, 9))
    assert sidelengths == [15, 11]


# 4 x 4 x 4 grid of blobs with 4 nm spacing
DENSE_BLOBS = np.stack(
    np.meshgrid(*[np.arange(4) * 4.0 - 6.0] * 3, indexing="ij"), axis=-1
).reshape(-1, 3)


# Neighboring molecules outside the padded cube are aliased into the cube, so the
# subtomograms are less consistent with the tomogram for smaller padding.
@pytest.mark.parametrize("padding, min_corr", [(2.0, 0.85), (3.0, 0.93)])
@pytest.mark.parametrize("tomo_hand", [1, -1])
def test_dense_molecules_consistent_with_tomogram_reconstruction(
    tmp_path, tomo_hand, padding, min_corr
):
    ts = _make_tilt_series(tmp_path / "blobs.mrc", DENSE_BLOBS)
    scale = 0.5
    shape = (60, 100, 100)
    tomo_center = np.array(shape) // 2 * scale
    rot = Rotation.random(len(DENSE_BLOBS), random_state=7)
    blobs = DENSE_BLOBS if tomo_hand == 1 else _mirror_z(DENSE_BLOBS)
    mole = Molecules(blobs + tomo_center, rot)
    kwargs = dict(output_shape=(9, 9, 9), scale=scale)
    loader = BackProjectionLoader(
        ts,
        mole,
        tomogram_center=tomo_center,
        tomo_hand=tomo_hand,
        padding=padding,
        output_shape=(9, 9, 9),
        binsize=scale / IMAGE_SCALE,
    )
    out = loader.asnumpy()
    # NOTE: the contrast of the tomogram is not inverted
    tomo = -loader.reconstruct_tomogram(shape, patch_size=32)
    ref = SubtomogramLoader(tomo, mole, **kwargs).asnumpy()
    for img0, img1 in zip(ref, out):
        assert np.corrcoef(img0.ravel(), img1.ravel())[0, 1] > min_corr


@pytest.mark.parametrize("corner_safe", [False, True])
def test_tomo_hand(tmp_path, corner_safe):
    # In the mirrored tomogram, blobs are at the local (1, 0, 1.5) nm position of the
    # molecules.
    rot = Rotation.random(len(BLOBS), random_state=8)
    offset = np.array([1.0, 0.0, 1.5])
    blobs = BLOBS + rot.apply(_mirror_z(offset))
    ts = _make_tilt_series(tmp_path / "hand.mrc", blobs)
    mole = Molecules(_mirror_z(BLOBS) + CENTER, _mirror_rotation(rot))
    kwargs = dict(output_shape=(13, 13, 13), corner_safe=corner_safe)
    loader = _loader(ts, mole, tomo_hand=-1, **kwargs)
    for subtomo in loader.asnumpy():
        assert_allclose(_center_of_mass(subtomo), [8, 6, 9], atol=0.15)


def test_properties(blob_tilt_series):
    mole = Molecules(BLOBS + CENTER)
    loader = _loader(blob_tilt_series, mole)
    assert isinstance(loader.tilt_model, SingleAxis)
    assert loader.tilt_model.tilt_range == (-40.0, 60.0)
    assert_allclose(loader.tomogram_center, CENTER)
    assert loader.tilt_series is blob_tilt_series
    binned = loader.binning(2, compute=False)
    # the pixel size is that of the tilt series by default
    assert loader.binsize == 1.0
    assert loader.scale == pytest.approx(blob_tilt_series.scale)
    assert binned.binsize == pytest.approx(2.0)
    assert binned.scale == pytest.approx(loader.scale * 2)
    assert loader.replace(scale=1.0).binsize == pytest.approx(1.0 / IMAGE_SCALE)
    assert loader.replace(binsize=3).binsize == 3
    assert loader.replace(binsize=3).scale == pytest.approx(IMAGE_SCALE * 3)
    assert binned.binning(1.5).binsize == 3.0  # exact, not through the scale
    with pytest.raises(TypeError):
        loader.replace(scale=1.0, binsize=2)
    assert_allclose(binned.molecules.pos, loader.molecules.pos)
    assert_allclose(binned.tomogram_center, CENTER)
    assert binned.output_shape == loader.output_shape
    assert binned.tilt_model is loader.tilt_model
    assert binned.batch_size is loader.batch_size is None
    assert binned.tomo_hand == loader.tomo_hand == 1
    assert binned.padding == loader.padding == 2.0
    assert binned.preprocessor == loader.preprocessor == TiltImagePreprocessor()
    assert _loader(blob_tilt_series, mole, padding=3).replace(scale=1.0).padding == 3.0
    mirrored = _loader(blob_tilt_series, mole, tomo_hand=-1)
    assert mirrored.tilt_model.tilt_range == (-60.0, 40.0)
    assert mirrored.binning(2).tomo_hand == -1
    repr(loader)


def test_groupby(blob_tilt_series):
    # `groupby` replaces the loaders by `scale`
    group = pl.Series("group", [0, 0, 1, 1, 1])
    mole = Molecules(BLOBS + CENTER).with_features(group)
    loader = _loader(blob_tilt_series, mole, output_shape=(7, 7, 7)).binning(2)
    out = {key: sub for key, sub in loader.groupby("group")}
    assert len(out) == 2
    for sub in out.values():
        assert isinstance(sub, BackProjectionLoader)
        assert sub.binsize == pytest.approx(2.0)
        assert sub.asnumpy().shape[1:] == (7, 7, 7)


def test_empty(blob_tilt_series):
    loader = _loader(blob_tilt_series, Molecules.empty())
    assert loader.construct_dask().shape == (0, 13, 13, 13)
    assert len(loader.construct_loading_tasks()) == 0


def test_invalid_input(blob_tilt_series):
    mole = Molecules(BLOBS + CENTER)
    with pytest.raises(TypeError):
        BackProjectionLoader(blob_tilt_series, BLOBS)
    with pytest.raises(ValueError):
        BackProjectionLoader(blob_tilt_series, mole, tomogram_center=(0, 0))
    with pytest.raises(ValueError):
        BackProjectionLoader(blob_tilt_series, mole, batch_size=0)
    with pytest.raises(ValueError):
        BackProjectionLoader(blob_tilt_series, mole, tomo_hand=0)
    with pytest.raises(ValueError):
        BackProjectionLoader(blob_tilt_series, mole, padding=0.5)
    with pytest.raises(ValueError):
        BackProjectionLoader(blob_tilt_series, mole, binsize=0)
    with pytest.raises(ValueError):
        BackProjectionLoader(blob_tilt_series, mole).binning(-2)
    with pytest.raises(TypeError):
        BackProjectionLoader(blob_tilt_series, mole, preprocess="yes")
    with pytest.raises(TypeError):
        BackProjectionLoader(object(), mole)
    with pytest.raises(ValueError):  # image_path is not set
        BackProjectionLoader(blob_tilt_series.replace(image_path=None), mole)
    with pytest.raises(ValueError):  # scale is not set
        BackProjectionLoader(blob_tilt_series.replace(scale=None), mole)


def _local_center_of_mass(tomo: np.ndarray, pos: np.ndarray, r: int = 4):
    corner = np.round(pos).astype(int) - r
    sl = tuple(slice(c, c + 2 * r + 1) for c in corner)
    return _center_of_mass(tomo[sl]) + corner


@pytest.mark.parametrize("scale", [0.5, 0.8])
def test_reconstruct_tomogram(blob_tilt_series, scale):
    shape = (30, 100, 110)
    loader = _loader(blob_tilt_series, Molecules(BLOBS + CENTER), scale=scale)
    tomo = loader.reconstruct_tomogram(shape, patch_size=16)
    assert tomo.shape == shape
    assert tomo.dtype == np.float16
    assert loader.reconstruct_tomogram(shape, dtype=np.float32).dtype == np.float32
    tomo = -tomo  # the contrast of the tomogram is not inverted
    # the tomogram is centered at the tomogram center
    for pos in BLOBS / scale + np.array(shape) // 2:
        assert_allclose(_local_center_of_mass(tomo, pos), pos, atol=0.15)


def test_reconstruct_tomogram_invalid_input(blob_tilt_series):
    loader = _loader(blob_tilt_series, Molecules(BLOBS + CENTER))
    with pytest.raises(ValueError):
        loader.reconstruct_tomogram((10, 10, 10), patch_size=0)
    with pytest.raises(ValueError):
        loader.reconstruct_tomogram((10, 0, 10))
    with pytest.raises(ValueError):
        loader.reconstruct_tomogram((10, 10, 10), blend_margin=-1)


@pytest.mark.parametrize("depth", [30, 31])
def test_reconstruct_tomogram_tomo_hand(blob_tilt_series, depth):
    shape = (depth, 100, 110)
    loader = _loader(blob_tilt_series, Molecules(BLOBS + CENTER), tomo_hand=-1)
    tomo = loader.reconstruct_tomogram(shape, patch_size=16)
    assert tomo.shape == shape
    tomo = -tomo  # the contrast of the tomogram is not inverted
    for pos in _mirror_z(BLOBS) / 0.5 + np.array(shape) // 2:
        assert_allclose(_local_center_of_mass(tomo, pos), pos, atol=0.15)


# asymmetric particle of 3 blobs, in nm
PARTICLE = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.5, 0.0, 2.0]])


def _particle_points(centers: np.ndarray, rot: Rotation, particle=PARTICLE):
    """Blob positions of the particles at the centers."""
    offsets = np.einsum("nij,kj->nki", rot.as_matrix(), particle)
    return (centers[:, None] + offsets).reshape(-1, 3)


def _particle_template(scale: float, size: int, particle=PARTICLE) -> np.ndarray:
    """Template of the particle with bright blobs, consistent with the tilt series."""
    sigma = 1.5 * IMAGE_SCALE / scale
    center = (size - 1) / 2
    inds = np.indices((size,) * 3, dtype=np.float32)
    template = np.zeros((size,) * 3, dtype=np.float32)
    for blob in particle / scale + center:
        dist2 = sum((ind - b) ** 2 for ind, b in zip(inds, blob))
        template += np.exp(-dist2 / (2 * sigma**2))
    return template


def _random_particles(n: int, seed: int) -> tuple[np.ndarray, Rotation]:
    """Random positions (relative to the tomogram center) and rotations."""
    rng = np.random.default_rng(seed)
    centers = np.stack(
        [rng.uniform(-4, 4, n), rng.uniform(-25, 25, n), rng.uniform(-28, 28, n)],
        axis=1,
    )
    return centers, Rotation.random(n, random_state=seed)


def _perturbed(ts: TiltSeries, amplitude: float, seed: int = 0) -> TiltSeries:
    """Copy of the tilt series with randomly shifted tilt images (nm)."""
    rng = np.random.default_rng(seed)
    shifts = rng.uniform(-amplitude, amplitude, size=ts.sample_translations.shape)
    return ts.replace(sample_translations=ts.sample_translations + shifts)


@pytest.mark.parametrize("tomo_hand", [1, -1])
@pytest.mark.parametrize("scale, size", [(0.5, 16), (1.0, 9)])
def test_refine_tilt_series(tmp_path, scale, size, tomo_hand):
    centers, rot = _random_particles(30, seed=1)
    ts = _make_tilt_series(tmp_path / "ts.mrc", _particle_points(centers, rot))
    if tomo_hand == 1:
        mole = Molecules(centers + CENTER, rot)
        template = _particle_template(scale, size)
    else:
        # particles in the mirrored tomogram are also mirrored
        mole = Molecules(_mirror_z(centers) + CENTER, _mirror_rotation(rot))
        template = _particle_template(scale, size, _mirror_z(PARTICLE))
    loader = _loader(_perturbed(ts, 1.2), mole, scale=scale, tomo_hand=tomo_hand)
    refined = loader.refine_tilt_series([(mole, template)], max_shift=2.0)
    error = refined.tilt_series.sample_translations - ts.sample_translations
    assert np.abs(error).max() < 0.1  # nm
    assert refined.tilt_series.local_shifts_2d is None


def test_refine_local_shifts(tmp_path):
    centers, rot = _random_particles(60, seed=2)
    rng = np.random.default_rng(3)
    coefs = rng.normal(scale=0.4, size=(TILT_ANGLES.size, 2, 3))
    half_extent = np.array(IMAGE_SHAPE) * IMAGE_SCALE / 2

    def field(projected):
        # bilinear deformation of the tilt images, up to ~1 nm
        y, x = np.moveaxis(projected / half_extent, -1, 0)
        terms = np.stack([y, x, y * x], axis=-1)[..., np.newaxis, :]  # (n, T, 1, 3)
        return (coefs * terms).sum(-1)

    ts = _make_tilt_series(
        tmp_path / "ts.mrc", _particle_points(centers, rot), local_shifts_2d=field
    )
    initial = ts.replace(local_shifts_2d=None)
    mole = Molecules(centers + CENTER, rot)
    loader = _loader(initial, mole)
    refined = loader.refine_tilt_series(
        [(mole, _particle_template(0.5, 16))], grid_shape=(2, 2)
    )
    assert isinstance(refined.tilt_series.local_shifts_2d, LocalShifts2D)
    true_pos = ts.project_points(centers)
    error_before = np.linalg.norm(initial.project_points(centers) - true_pos, axis=-1)
    error_after = np.linalg.norm(
        refined.tilt_series.project_points(centers) - true_pos, axis=-1
    )
    assert error_before.mean() > 0.25  # nm
    assert error_after.mean() < 0.05
    # the refined tilt series can be used for reconstruction
    assert refined.reconstruct_tomogram((16, 40, 40), patch_size=16).shape == (
        16,
        40,
        40,
    )


def test_refine_with_local_alignment(tmp_path):
    centers, rot = _random_particles(30, seed=8)
    rng = np.random.default_rng(9)
    ntilts = TILT_ANGLES.size
    local = LocalAlignment(
        origin=[-20.0, -25.0],
        spacing=[40.0, 50.0],
        tilt_angles=rng.normal(scale=1.0, size=(2, 2, ntilts)),
        translations=rng.normal(scale=0.6, size=(2, 2, ntilts, 2)),
    )
    ts = _make_tilt_series(
        tmp_path / "ts.mrc", _particle_points(centers, rot), local_shifts_2d=local
    )
    mole = Molecules(centers + CENTER, rot)
    template = _particle_template(0.5, 16)
    initial = _perturbed(ts, 1.2)  # with the local alignment
    true_pos = ts.project_points(centers)

    refined = _loader(initial, mole).refine_tilt_series([(mole, template)])
    assert refined.tilt_series.local_shifts_2d is local
    error = refined.tilt_series.sample_translations - ts.sample_translations
    assert np.abs(error).max() < 0.1

    # the local alignment is the base of the refined local shifts
    refined = _loader(initial, mole).refine_tilt_series(
        [(mole, template)], grid_shape=(2, 2)
    )
    model = refined.tilt_series.local_shifts_2d
    assert isinstance(model, LocalShifts2D) and model.base is local
    error = np.linalg.norm(
        refined.tilt_series.project_points(centers) - true_pos, axis=-1
    )
    assert error.mean() < 0.05

    # the local alignment cannot be fully refined by the uniform shifts
    without = _loader(initial.replace(local_shifts_2d=None), mole)
    refined = without.refine_tilt_series([(mole, template)])
    error = np.linalg.norm(
        refined.tilt_series.project_points(centers) - true_pos, axis=-1
    )
    assert error.mean() > 0.2


def test_refine_with_multiple_references(tmp_path):
    other = np.array([[0.0, -1.5, 1.5], [0.0, 1.5, -1.5]])  # another kind of particle
    centers0, rot0 = _random_particles(15, seed=4)
    centers1, rot1 = _random_particles(15, seed=5)
    points = np.concatenate(
        [_particle_points(centers0, rot0), _particle_points(centers1, rot1, other)]
    )
    ts = _make_tilt_series(tmp_path / "ts.mrc", points)
    mole0 = Molecules(centers0 + CENTER, rot0)
    mole1 = Molecules(centers1 + CENTER, rot1)
    loader = _loader(_perturbed(ts, 1.2), mole0)
    refined = loader.refine_tilt_series(
        [
            (mole0, _particle_template(0.5, 16)),
            (mole1, _particle_template(0.5, 16, other)),
        ]
    )
    error = refined.tilt_series.sample_translations - ts.sample_translations
    assert np.abs(error).max() < 0.1


def test_refined_loader(tmp_path):
    centers, rot = _random_particles(20, seed=6)
    ts = _make_tilt_series(tmp_path / "ts.mrc", _particle_points(centers, rot))
    mole = Molecules(centers + CENTER, rot)
    initial = _perturbed(ts, 1.5)
    translations = initial.sample_translations.copy()
    loader = _loader(initial, mole, output_shape=(15, 15, 15))
    template = _particle_template(0.5, 15)
    with loader.cache_tilt_images():
        images = loader._images.get(1)[1].copy()
        refined = loader.refine_tilt_series([(mole, template)])
        # the cache is not modified, and shared with the refined loader
        assert_allclose(loader._images.get(1)[1], images)
        assert refined._images.get(1)[1] is loader._images.get(1)[1]
    assert isinstance(refined, BackProjectionLoader)
    assert refined.tilt_series is not initial
    assert loader.tilt_series is initial
    assert_allclose(initial.sample_translations, translations)  # not modified
    assert refined.molecules is loader.molecules

    # subtomograms are sharper after refinement
    def _score(ldr: BackProjectionLoader) -> float:
        avg = ldr.average()
        return np.corrcoef(avg.ravel(), template.ravel())[0, 1]

    assert _score(refined) > _score(loader) + 0.05


def test_with_tilt_series(blob_tilt_series):
    loader = _loader(blob_tilt_series, Molecules(BLOBS + CENTER), scale=1.0)
    shifted = blob_tilt_series.replace(
        sample_translations=blob_tilt_series.sample_translations + 1.0
    )
    with loader.cache_tilt_images():
        images = loader._images.get(2)[1]
        other = loader.with_tilt_series(shifted)
        assert other.tilt_series is shifted
        assert other.binsize == loader.binsize
        assert other.molecules is loader.molecules
        # the preprocessed images are shared
        assert other._images.get(2)[1] is images
    # the images of another pixel size are not shared
    another = loader.with_tilt_series(blob_tilt_series.replace(scale=0.6))
    assert another._images.cache is not loader._images.cache


def test_refine_invalid_input(blob_tilt_series):
    loader = _loader(blob_tilt_series, Molecules(BLOBS + CENTER))
    template = _particle_template(0.5, 16)
    with pytest.raises(ValueError):
        loader.refine_tilt_series([])
    with pytest.raises(ValueError):
        loader.refine_tilt_series([(Molecules.empty(), template)])
    with pytest.raises(ValueError):
        _loader(
            blob_tilt_series, Molecules(BLOBS + CENTER), tomo_hand=-1
        ).refine_tilt_series([(Molecules.empty(), template)])
    with pytest.raises(ValueError):
        loader.refine_tilt_series(
            [(Molecules(BLOBS + CENTER), template)], grid_shape=(0, 2)
        )


@pytest.mark.parametrize(
    "kwargs", [{}, {"low": 0.02, "high": 0.4}, {"subtract_background": False}]
)
def test_tilt_by_tilt_preprocessing(blob_tilt_series, kwargs):
    from acryo.loader._back_projection import _TiltSeriesImages

    preprocessor = TiltImagePreprocessor(**kwargs)
    raw = mrcfile.read(blob_tilt_series.image_path)
    ref = np.stack([preprocessor(image) for image in raw])
    _, images = _TiltSeriesImages(blob_tilt_series, preprocessor).get(1)
    assert_allclose(images, ref, atol=1e-5)
    # cached in memory, and read-only because the cache is shared
    assert type(images) is np.ndarray
    assert images.dtype == np.float32
    assert not images.flags.writeable


def test_preprocess_option(blob_tilt_series):
    mole = Molecules(BLOBS + CENTER)
    raw = mrcfile.read(blob_tilt_series.image_path)
    loader = _loader(blob_tilt_series, mole, preprocess=False)
    assert loader.preprocessor is None
    assert_allclose(loader._images.get(1)[1], raw)
    preprocessor = TiltImagePreprocessor(low=0.01, padding=16)
    loader = _loader(blob_tilt_series, mole, preprocess=preprocessor)
    assert loader.preprocessor is preprocessor
    assert loader.replace(scale=1.0).preprocessor is preprocessor
    assert_allclose(loader._images.get(1)[1][3], preprocessor(raw[3]), atol=1e-5)


def test_filters_receive_tilt_index_and_frequency(blob_tilt_series):
    calls: list[tuple[int, float]] = []

    def _filter(freq_y, freq_x, index):
        calls.append((index, float(np.abs(freq_x).max())))
        return 1.0

    preprocessor = TiltImagePreprocessor().with_filters(_filter)
    loader = _loader(
        blob_tilt_series, Molecules(BLOBS + CENTER), preprocess=preprocessor
    )
    loader.binning(2).average()
    # tilt images are preprocessed in parallel
    assert sorted(index for index, _ in calls) == list(range(TILT_ANGLES.size))
    # frequencies in 1/nm, up to the Nyquist frequency of the binned images
    assert_allclose([freq for _, freq in calls], 0.5 / (2 * IMAGE_SCALE))


def test_ctf_correction_of_tilt_series(blob_tilt_series):
    from acryo.ctf import CTFModel, PatchCTFCorrection, TiltSeriesCTF

    ctf = TiltSeriesCTF(defocus_u=np.linspace(-2.0, -3.0, TILT_ANGLES.size))
    ts = blob_tilt_series.replace(ctf=ctf)
    mole = Molecules(BLOBS + CENTER)
    loader = _loader(ts, mole)
    (correction,) = loader.preprocessor.filters
    assert isinstance(correction, PatchCTFCorrection)
    # the tilt images are preprocessed with the CTF correction
    ref = TiltImagePreprocessor().with_filters(ts.ctf_correction())
    raw = mrcfile.read(ts.image_path)
    assert_allclose(
        loader._images.get(1)[1][3],
        ref(raw[3], index=3, scale=IMAGE_SCALE),
        atol=1e-5,
    )
    assert not np.allclose(loader._images.get(1)[1][3], TiltImagePreprocessor()(raw[3]))
    # the CTF is not corrected twice
    custom = TiltImagePreprocessor(low=0.01).with_ctf(CTFModel.from_kv(300, 2.7))
    assert _loader(ts, mole, preprocess=custom).preprocessor is custom
    assert _loader(ts, mole, preprocess=False).preprocessor is None
    # the CTF of the remaining tilts
    excluded = loader.exclude_tilts(EXCLUDED)
    (correction,) = excluded.preprocessor.filters
    assert_allclose(correction.ctf.defocus_u, ctf.defocus_u[KEPT])
    # the CTF correction is created from the new tilt series
    other = loader.with_tilt_series(ts.replace(scale=0.6, ctf=None))
    assert other.preprocessor.filters == ()


def test_filters_are_applied_to_the_right_tilts(blob_tilt_series):
    plain = TiltImagePreprocessor(normalize=False)
    scaled = plain.with_filters(lambda fy, fx, index: float(index + 1))
    mole = Molecules(BLOBS + CENTER)
    ref = _loader(blob_tilt_series, mole, preprocess=plain)._images.get(2)[1]
    out = _loader(blob_tilt_series, mole, preprocess=scaled)._images.get(2)[1]
    factors = np.arange(1, TILT_ANGLES.size + 1)[:, np.newaxis, np.newaxis]
    assert_allclose(out, ref * factors, rtol=1e-4, atol=1e-4)


def test_reconstruct_tomogram_with_filters(blob_tilt_series):
    # the tomogram is reconstructed from the preprocessed images
    preprocessor = TiltImagePreprocessor(normalize=False)
    inverted = preprocessor.with_filters(lambda fy, fx, i: -1.0)
    mole = Molecules(BLOBS + CENTER)
    shape = (20, 40, 40)
    ref = _loader(blob_tilt_series, mole, preprocess=preprocessor)
    loader = _loader(blob_tilt_series, mole, preprocess=inverted)
    assert_allclose(
        loader.reconstruct_tomogram(shape, patch_size=16, dtype=np.float32),
        -ref.reconstruct_tomogram(shape, patch_size=16, dtype=np.float32),
        atol=1e-4,
    )


@pytest.mark.parametrize("tomo_hand", [1, -1])
def test_reconstruct_tomogram_save_path(blob_tilt_series, tmp_path, tomo_hand):
    loader = _loader(blob_tilt_series, Molecules(BLOBS + CENTER), tomo_hand=tomo_hand)
    shape = (21, 50, 60)
    ref = loader.reconstruct_tomogram(shape, patch_size=16, dtype=np.float32)
    path = tmp_path / "tomogram.mrc"
    out = loader.reconstruct_tomogram(shape, patch_size=16, save_path=path)
    assert isinstance(out, np.memmap)
    assert out.dtype == np.float16
    assert_allclose(out, ref, rtol=2e-3, atol=1e-4)
    with mrcfile.open(path) as mrc:
        assert mrc.header.mode == 12
        assert_allclose(mrc.data, ref, rtol=2e-3, atol=1e-4)
        assert mrc.voxel_size.x == pytest.approx(loader.scale * 10)
    del out


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.uint16, np.float16])
@pytest.mark.parametrize("extended_header", [False, True])
def test_mrc_images_are_read_one_by_one(tmp_path, monkeypatch, dtype, extended_header):
    from acryo.projection._io import MrcImageStack

    rng = np.random.default_rng(0)
    data = (rng.normal(size=(5, 12, 17)) * 100).astype(dtype)
    path = tmp_path / "stack.mrc"
    with mrcfile.new(path) as mrc:
        mrc.set_data(data)
        if extended_header:
            mrc.set_extended_header(np.zeros(10, dtype=[("a", "<i4", (7,))]))

    # the whole file must not be read nor memory-mapped
    def _fail(*args, **kwargs):
        raise AssertionError("The whole file is read.")

    monkeypatch.setattr(mrcfile, "mmap", _fail)
    monkeypatch.setattr(mrcfile, "read", _fail)
    monkeypatch.setattr(np, "memmap", _fail)
    stack = MrcImageStack(path)
    assert stack.shape == data.shape
    images = list(stack.iter_images([3, 0, -1]))
    for image, index in zip(images, [3, 0, -1]):
        assert image.dtype == np.float32
        assert_allclose(image, data[index].astype(np.float32))
    with pytest.raises(IndexError):
        list(stack.iter_images([5]))


def test_tilt_images_are_loaded_once_per_binsize(blob_tilt_series, monkeypatch):
    from acryo.loader._back_projection import _TiltSeriesImages

    loaded: list[int] = []
    _load = _TiltSeriesImages._load

    def _recorded_load(self, binsize):
        loaded.append(binsize)
        return _load(self, binsize)

    monkeypatch.setattr(_TiltSeriesImages, "_load", _recorded_load)
    loader = _loader(blob_tilt_series, Molecules(BLOBS + CENTER))
    assert loaded == []  # lazy
    with loader.cache_tilt_images() as out:
        assert out is loader
        loader.average()
        loader.head(3).average()
        loader.binning(2).replace(output_shape=(7, 7, 7)).average()
        loader.binning(2).average(output_shape=(7, 7, 7))
        loader.replace(binsize=1.4).average()  # binsize 1 is used again
        assert loaded == [1, 2]
    # the cache is released
    assert loader._images.cache.images == {}
    assert loader._images.cache.reconstructors == {}
    loader.average()
    assert loaded == [1, 2, 1]


def test_tilt_images_are_not_cached_by_default(blob_tilt_series, monkeypatch):
    from acryo.loader._back_projection import _TiltSeriesImages

    loaded: list[int] = []
    _load = _TiltSeriesImages._load

    def _recorded_load(self, binsize):
        loaded.append(binsize)
        return _load(self, binsize)

    monkeypatch.setattr(_TiltSeriesImages, "_load", _recorded_load)
    loader = _loader(blob_tilt_series, Molecules(BLOBS + CENTER))
    loader.average()
    loader.head(3).average()
    assert loaded == [1, 1]
    assert loader._images.cache.images == {}


def test_cache_is_shared_and_nested(blob_tilt_series, monkeypatch):
    from acryo.loader._back_projection import _TiltSeriesImages

    loaded: list[int] = []
    _load = _TiltSeriesImages._load

    def _recorded_load(self, binsize):
        loaded.append(binsize)
        return _load(self, binsize)

    monkeypatch.setattr(_TiltSeriesImages, "_load", _recorded_load)
    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=12))
    loader = _loader(blob_tilt_series, mole)
    derived = loader.replace(output_shape=(9, 9, 9))
    with loader.cache_tilt_images():
        with derived.cache_tilt_images():
            ref = derived.asnumpy()
        # the cache is kept until the outermost context exits
        assert set(loader._images.cache.images) == {(1, None)}
        loader.asnumpy()
        # created in the context, computed after the context
        stack = derived.construct_dask()
    assert loaded == [1]
    assert loader._images.cache.images == {}
    assert_allclose(stack.compute(), ref)
    assert loaded == [1]
    # the cache is released even if an exception is raised
    with pytest.raises(RuntimeError):
        with loader.cache_tilt_images():
            loader.asnumpy()
            raise RuntimeError
    assert loader._images.cache.images == {}
    assert not loader._images.cache.enabled


def test_image_indices(tmp_path, blob_tilt_series):
    # the same tilt images with an excluded image at the beginning
    with mrcfile.open(blob_tilt_series.image_path) as mrc:
        data = mrc.data.copy()
    path = tmp_path / "with_excluded.mrc"
    with mrcfile.new(path) as mrc:
        mrc.set_data(np.concatenate([np.ones_like(data[:1]), data]))
    ts = blob_tilt_series.replace(
        image_path=path, image_indices=np.arange(1, data.shape[0] + 1)
    )
    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=9))
    ref = _loader(blob_tilt_series, mole).asnumpy()
    assert_allclose(_loader(ts, mole).asnumpy(), ref, atol=1e-5)


EXCLUDED = [0, 5, TILT_ANGLES.size - 1]
KEPT = np.setdiff1d(np.arange(TILT_ANGLES.size), EXCLUDED)


@pytest.mark.parametrize("tomo_hand", [1, -1])
def test_exclude_tilts(tmp_path, blob_tilt_series, tomo_hand):
    # the tilt series of the kept tilts, made from scratch
    raw = mrcfile.read(blob_tilt_series.image_path)
    path = tmp_path / "kept.mrc"
    with mrcfile.new(path) as mrc:
        mrc.set_data(raw[KEPT])
    ts = blob_tilt_series
    ref_ts = TiltSeries(
        tilt_angles=ts.tilt_angles[KEPT],
        tilt_axis_angle=ts.tilt_axis_angle[KEPT],
        sample_translations=ts.sample_translations[KEPT],
        scale=IMAGE_SCALE,
        image_path=path,
    )
    rot = Rotation.random(len(BLOBS), random_state=13)
    blobs = BLOBS if tomo_hand == 1 else _mirror_z(BLOBS)
    mole = Molecules(blobs + CENTER, rot)
    loader = _loader(ts, mole, tomo_hand=tomo_hand)
    out = loader.exclude_tilts(EXCLUDED)
    assert isinstance(out, BackProjectionLoader)
    assert out.tilt_series.ntilts == KEPT.size
    assert_allclose(out.tilt_series.image_indices, KEPT)
    assert out.molecules is loader.molecules
    assert out.tomo_hand == tomo_hand
    assert loader.tilt_series is ts  # not modified
    ref = _loader(ref_ts, mole, tomo_hand=tomo_hand)
    assert_allclose(out.asnumpy(), ref.asnumpy(), atol=1e-5)
    # the default tilt model is created from the remaining tilt angles
    assert out.tilt_model.tilt_range == ref.tilt_model.tilt_range
    assert out.tilt_model.tilt_range != loader.tilt_model.tilt_range
    derived = loader.replace(output_shape=(9, 9, 9)).exclude_tilts(EXCLUDED)
    assert derived.tilt_model.tilt_range == ref.tilt_model.tilt_range
    # the indices are of the tilt series of the loader
    twice = loader.exclude_tilts(0).exclude_tilts([4, -1])
    assert_allclose(twice.tilt_series.image_indices, KEPT)
    assert_allclose(twice.asnumpy(), out.asnumpy(), atol=1e-6)


def test_exclude_tilts_keeps_given_tilt_model(blob_tilt_series):
    tilt_model = single_axis((-60.0, 60.0), axis="y")
    loader = _loader(blob_tilt_series, Molecules(BLOBS + CENTER), tilt_model=tilt_model)
    assert loader.exclude_tilts(EXCLUDED).tilt_model is tilt_model
    assert loader.replace().exclude_tilts(EXCLUDED).tilt_model is tilt_model


def test_exclude_tilts_with_filters(blob_tilt_series):
    plain = TiltImagePreprocessor(normalize=False)
    scaled = plain.with_filters(lambda fy, fx, index: float(index + 1))
    mole = Molecules(BLOBS + CENTER)
    ref = _loader(blob_tilt_series, mole, preprocess=plain)._images.get(2)[1]
    loader = _loader(blob_tilt_series, mole, preprocess=scaled).exclude_tilts(EXCLUDED)
    # the filters are applied to the same tilts as before
    factors = (KEPT + 1)[:, np.newaxis, np.newaxis]
    assert_allclose(loader._images.get(2)[1], ref[KEPT] * factors, rtol=1e-4, atol=1e-4)
    # the preprocessor is consistent with the tilt series
    same = _loader(loader.tilt_series, mole, preprocess=loader.preprocessor)
    assert_allclose(same._images.get(2)[1], loader._images.get(2)[1])
    # without preprocessing
    raw = mrcfile.read(blob_tilt_series.image_path)
    loader = _loader(blob_tilt_series, mole, preprocess=False).exclude_tilts(EXCLUDED)
    assert loader.preprocessor is None
    assert_allclose(loader._images.get(1)[1], raw[KEPT])


def test_exclude_tilts_cache(blob_tilt_series, monkeypatch):
    from acryo.loader._back_projection import _TiltSeriesImages
    from acryo.projection._io import MrcImageStack

    loaded: list[tuple[int, int]] = []
    read: list[int] = []
    _load = _TiltSeriesImages._load
    _iter_images = MrcImageStack.iter_images

    def _recorded_load(self, binsize):
        loaded.append((self.tilt_series.ntilts, binsize))
        return _load(self, binsize)

    def _recorded_iter_images(self, indices):
        indices = list(indices)
        read.extend(indices)
        return _iter_images(self, indices)

    monkeypatch.setattr(_TiltSeriesImages, "_load", _recorded_load)
    monkeypatch.setattr(MrcImageStack, "iter_images", _recorded_iter_images)
    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=14))
    loader = _loader(blob_tilt_series, mole)
    ntilts = TILT_ANGLES.size
    with loader.cache_tilt_images():
        images = loader._images.get(1)[1]
        out = loader.exclude_tilts(EXCLUDED)
        # cropped from the cached images of all the tilts
        assert_allclose(out._images.get(1)[1], images[KEPT])
        out.average()
        out.exclude_tilts([0]).average()
        assert loaded == [(ntilts, 1)]
        # only the images of the kept tilts are loaded and cached
        read.clear()
        out.binning(2).average()
        out.binning(2).average()
        assert loaded == [(ntilts, 1), (KEPT.size, 2)]
        assert read == KEPT.tolist()
    assert loader._images.cache.images == {}
    # the images of the excluded tilts are not read
    read.clear()
    loader.exclude_tilts(EXCLUDED).average()
    assert read == KEPT.tolist()


def test_exclude_tilts_invalid_input(blob_tilt_series):
    loader = _loader(blob_tilt_series, Molecules(BLOBS + CENTER))
    assert loader.exclude_tilts(3).tilt_series.ntilts == TILT_ANGLES.size - 1
    assert loader.exclude_tilts([]).tilt_series is loader.tilt_series
    with pytest.raises(ValueError):
        loader.exclude_tilts(np.arange(TILT_ANGLES.size))
    with pytest.raises(IndexError):
        loader.exclude_tilts([TILT_ANGLES.size])
    with pytest.raises(TypeError):
        loader.exclude_tilts([0.5])


@pytest.mark.parametrize("scale", [1.0, 1.5])  # binsize 2 and 3
def test_binned_tilt_images(blob_tilt_series, monkeypatch, scale):
    from acryo.loader._back_projection import _TiltSeriesImages

    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=10))
    loader = _loader(blob_tilt_series, mole, scale=scale, output_shape=(9, 9, 9))
    assert loader._images.binsize_for(loader.binsize) == round(scale * 2)
    binned = loader.asnumpy()
    monkeypatch.setattr(_TiltSeriesImages, "binsize_for", lambda self, binsize: 1)
    unbinned = loader.replace().asnumpy()
    for img0, img1 in zip(binned, unbinned):
        assert np.corrcoef(img0.ravel(), img1.ravel())[0, 1] > 0.95
        assert_allclose(_center_of_mass(img0), [4, 4, 4], atol=0.15)


def test_subtomograms_are_not_kept(blob_tilt_series, recorded_calls):
    # the same subtomograms are reconstructed again in another computation
    mole = Molecules(BLOBS + CENTER)
    stack = _loader(blob_tilt_series, mole, batch_size=2).construct_dask()
    stack[:2].compute()
    assert [n for n, _ in recorded_calls] == [2]
    stack.compute()
    assert sorted(n for n, _ in recorded_calls) == [1, 2, 2, 2]


def test_batch_loader(blob_tilt_series):
    # tilt series loaders and a tomogram in a batch loader
    rng = np.random.default_rng(0)
    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=11))
    loader0 = _loader(blob_tilt_series, mole[:3])
    loader1 = _loader(blob_tilt_series, mole[3:], tomo_hand=-1)
    tomo = rng.normal(size=(40, 120, 140)).astype(np.float32)
    loader2 = SubtomogramLoader(
        tomo, mole, scale=IMAGE_SCALE, output_shape=(13, 13, 13)
    )
    batch = BatchLoader.from_loaders(
        [loader0, loader1, loader2], scale=IMAGE_SCALE, output_shape=(13, 13, 13)
    )
    repr(batch)
    assert list(batch.images) == [2]
    assert batch.tilt_models[0] is loader0.tilt_model
    assert batch.tilt_models[1] is loader1.tilt_model
    assert len(batch.loaders) == 3
    subs = dict(batch.loaders.iter_loaders())
    for i, ref in enumerate([loader0, loader1]):
        assert isinstance(subs[i], BackProjectionLoader)
        assert subs[i].binsize == 1.0  # exact, not through the scale
        assert subs[i].tomo_hand == ref.tomo_hand
    assert isinstance(subs[2], SubtomogramLoader)
    refs = [loader0.asnumpy(), loader1.asnumpy(), loader2.asnumpy()]
    # rotations are slightly changed when molecules are added to a batch loader
    assert_allclose(batch.asnumpy(), np.concatenate(refs), atol=1e-4)
    assert_allclose(
        batch.average(), np.concatenate(refs).mean(axis=0), rtol=1e-4, atol=1e-4
    )

    # the output shape of the loader is used if not set
    batch_unset = BatchLoader(scale=IMAGE_SCALE).add_loader(loader0)
    assert batch_unset.loaders[0].output_shape == (13, 13, 13)

    # the scale of the batch loader is converted to the binsize
    rescaled = batch.replace(scale=1.0)
    assert rescaled.loaders[0].binsize == pytest.approx(2.0)
    binned = batch.binning(2)
    assert binned.scale == pytest.approx(IMAGE_SCALE * 2)
    subs = dict(binned.loaders.iter_loaders())
    assert subs[0].binsize == subs[1].binsize == 2.0
    # only the molecules in the tomogram are translated by binning
    assert_allclose(subs[0].molecules.pos, loader0.molecules.pos, atol=1e-5)
    assert_allclose(subs[1].molecules.pos, loader1.molecules.pos, atol=1e-5)
    assert_allclose(
        subs[2].molecules.pos, loader2.molecules.pos - IMAGE_SCALE / 2, atol=1e-5
    )
    assert subs[2].image.shape == (20, 60, 70)
    assert binned.loaders[0].asnumpy().shape == (3, 13, 13, 13)

    filtered = batch.filter(pl.col("image-id") != 0)
    assert set(filtered.tilt_models) == {1, 2}
    assert len(filtered.loaders) == 2
    out = {key: sub for key, sub in batch.groupby("image-id")}
    assert isinstance(out[0].loaders[0], BackProjectionLoader)

    # molecules of the tilt series and the tomogram are interleaved
    order = [5, 0, 3, 6, 1, 7, 2, 4, 8, 9]
    shuffled = batch.replace(molecules=batch.molecules.subset(order))
    assert_allclose(shuffled.asnumpy(), np.concatenate(refs)[order], atol=1e-4)

    optimized = batch.order_optimize()
    assert_allclose(optimized.order_restore().molecules.pos, batch.molecules.pos)

    template = refs[0].mean(axis=0)
    aligned = batch.align(template, max_shifts=1.0)
    assert isinstance(aligned.loaders[1], BackProjectionLoader)
    assert_allclose(
        aligned.loaders[1].molecules.pos,
        loader1.align(template, max_shifts=1.0).molecules.pos,
        atol=1e-4,
    )


def test_batch_loader_image_id(blob_tilt_series):
    mole = Molecules(BLOBS + CENTER)
    loader0 = _loader(blob_tilt_series, mole[:3])
    loader1 = _loader(blob_tilt_series, mole[3:])
    # molecules of the same tilt series are added with the same image ID
    batch = BatchLoader(scale=IMAGE_SCALE)
    batch.add_loader(loader0, image_id="a").add_loader(loader1, image_id="a")
    assert len(batch.loaders) == 1
    assert batch.loaders["a"].molecules.count() == 5
    with pytest.raises(ValueError):
        batch.add_tomogram(np.zeros((4, 4, 4)), mole, image_id="a")
    batch.add_tomogram(np.zeros((4, 4, 4)), mole, image_id="b")
    with pytest.raises(ValueError):
        batch.add_loader(loader0, image_id="b")
    with pytest.raises(TypeError):
        batch.add_loader(BatchLoader(), image_id="c")

    # nested loaders are added one by one
    nested = BatchLoader(scale=IMAGE_SCALE)
    nested.add_loader(batch).add_loader(BatchLoader().add_loader(loader0))
    assert len(nested.loaders) == 3
    assert nested.molecules.count() == 13


@pytest.fixture
def live_tilt_images(monkeypatch) -> dict[str, int]:
    """Record the number of loads and of the loaded tilt images in memory."""
    import threading
    import weakref
    from acryo.loader._back_projection import _TiltSeriesImages

    state = {"loads": 0, "live": 0, "peak": 0}
    lock = threading.Lock()
    _load = _TiltSeriesImages._load

    def _release():
        with lock:
            state["live"] -= 1

    def _recorded_load(self, binsize):
        out = _load(self, binsize)
        with lock:
            state["loads"] += 1
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
        weakref.finalize(out, _release)
        return out

    monkeypatch.setattr(_TiltSeriesImages, "_load", _recorded_load)
    return state


def test_tilt_images_are_loaded_in_computation(blob_tilt_series, live_tilt_images):
    import gc

    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=13))
    loader = _loader(blob_tilt_series, mole, batch_size=2)
    stack = loader.construct_dask()
    assert live_tilt_images["loads"] == 0
    ref = stack.compute()
    # loaded once for all the tasks, and released after the computation
    assert live_tilt_images["loads"] == 1
    assert live_tilt_images["live"] == 0
    # a part of the array keeps the tilt images until the rest are computed
    assert_allclose(stack[:2].compute(), ref[:2])
    assert live_tilt_images["live"] == 1
    assert_allclose(stack[2:].compute(), ref[2:])
    assert live_tilt_images["loads"] == 2
    assert live_tilt_images["live"] == 0
    assert_allclose(stack.compute(), ref)
    assert live_tilt_images["loads"] == 3

    # each loading task does not load the tilt images again
    tasks = loader.construct_loading_tasks()
    assert live_tilt_images["loads"] == 4
    each = np.stack([task.compute() for task in tasks])
    assert_allclose(each, ref, rtol=1e-6, atol=1e-6)
    assert_allclose(np.stack(list(loader.load_iter())), ref, rtol=1e-6, atol=1e-6)
    assert live_tilt_images["loads"] == 5
    del tasks
    gc.collect()  # the graphs of the arrays have reference cycles
    assert live_tilt_images["live"] == 0


def test_tilt_images_of_batch_are_loaded_one_by_one(tmp_path, live_tilt_images):
    import dask

    rot = Rotation.random(len(BLOBS) * 4, random_state=14)
    mole = Molecules(np.tile(BLOBS, (4, 1)) + CENTER, rot)
    loaders = [
        _loader(_make_tilt_series(tmp_path / f"ts-{i}.mrc", BLOBS), mole, order=1)
        for i in range(4)
    ]
    batch = BatchLoader.from_loaders(loaders, scale=IMAGE_SCALE, order=1)
    template = loaders[0].average()
    live_tilt_images.update(loads=0, peak=0)
    with dask.config.set(scheduler="threads", num_workers=8):
        aligned = batch.align(template, max_shifts=0.5)
    assert live_tilt_images["loads"] == 4
    # the tilt images of the first loader is released before those of the last one
    # are loaded
    assert live_tilt_images["peak"] < 4
    assert live_tilt_images["live"] == 0
    assert_allclose(
        aligned.loaders[2].molecules.pos,
        loaders[2].align(template, max_shifts=0.5).molecules.pos,
        atol=1e-4,
    )
