import copy

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial.transform import Rotation

from acryo import Molecules, SubtomogramLoader, PseudoSubtomogramLoader
from acryo.backend import Backend
from acryo.tilt import SingleAxis

pytest.importorskip("torch_reconstruct_tomogram")

import torch  # noqa: E402
import mrcfile  # noqa: E402
from torch_tilt_series import TiltSeries  # noqa: E402

PIXEL_SPACING = 5.0  # Angstrom
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
        tilt_axis_angle=np.full(ntilts, 85.0),
        sample_translations=rng.normal(scale=10.0, size=(ntilts, 2)),
        pixel_spacing=PIXEL_SPACING,
        image_path=path,
        local_shifts_2d=local_shifts_2d,
    )
    projected = ts.project_points(torch.as_tensor(points_nm * 10)) / PIXEL_SPACING
    projected = projected.numpy() + np.array(IMAGE_SHAPE) // 2
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


def _loader(ts: TiltSeries, mole: Molecules, **kwargs) -> PseudoSubtomogramLoader:
    kwargs.setdefault("output_shape", (13, 13, 13))
    kwargs.setdefault("scale", PIXEL_SPACING / 10)
    return PseudoSubtomogramLoader(ts, mole, tomogram_center=CENTER, **kwargs)


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
    weight = np.clip(img, 0, None) ** 4
    inds = np.indices(img.shape).reshape(3, -1)
    return (inds * weight.ravel()).sum(axis=1) / weight.sum()


# NOTE: scales are chosen so that the crop sizes of the tilt images in
# torch-reconstruct-tomogram are both even and odd.
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
    loader = PseudoSubtomogramLoader(
        blob_tilt_series,
        mole,
        tomogram_center=tomo_center,
        tomo_hand=tomo_hand,
        **kwargs,
    )
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
    loader = PseudoSubtomogramLoader(
        ts,
        mole,
        tomogram_center=tomo_center,
        tomo_hand=tomo_hand,
        padding=padding,
        **kwargs,
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
    assert binned.scale == pytest.approx(loader.scale * 2)
    assert_allclose(binned.molecules.pos, loader.molecules.pos)
    assert_allclose(binned.tomogram_center, CENTER)
    assert binned.output_shape == loader.output_shape
    assert binned.tilt_model is loader.tilt_model
    assert binned.batch_size is loader.batch_size is None
    assert binned.tomo_hand == loader.tomo_hand == 1
    assert binned.padding == loader.padding == 2.0
    assert _loader(blob_tilt_series, mole, padding=3).replace(scale=1.0).padding == 3.0
    mirrored = _loader(blob_tilt_series, mole, tomo_hand=-1)
    assert mirrored.tilt_model.tilt_range == (-60.0, 40.0)
    assert mirrored.binning(2).tomo_hand == -1
    repr(loader)


def test_empty(blob_tilt_series):
    loader = _loader(blob_tilt_series, Molecules.empty())
    assert loader.construct_dask().shape == (0, 13, 13, 13)
    assert len(loader.construct_loading_tasks()) == 0


def test_invalid_input(blob_tilt_series):
    mole = Molecules(BLOBS + CENTER)
    with pytest.raises(TypeError):
        PseudoSubtomogramLoader(blob_tilt_series, BLOBS)
    with pytest.raises(ValueError):
        PseudoSubtomogramLoader(blob_tilt_series, mole, tomogram_center=(0, 0))
    with pytest.raises(ValueError):
        PseudoSubtomogramLoader(blob_tilt_series, mole, batch_size=0)
    with pytest.raises(ValueError):
        PseudoSubtomogramLoader(blob_tilt_series, mole, tomo_hand=0)
    with pytest.raises(ValueError):
        PseudoSubtomogramLoader(blob_tilt_series, mole, padding=0.5)
    ts = TiltSeries(
        tilt_angles=TILT_ANGLES,
        tilt_axis_angle=np.zeros_like(TILT_ANGLES),
        sample_translations=np.zeros((TILT_ANGLES.size, 2)),
        pixel_spacing=PIXEL_SPACING,
    )
    with pytest.raises(ValueError):
        PseudoSubtomogramLoader(ts, mole)  # image_path is not set


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
    assert tomo.dtype == np.float32
    tomo = -tomo  # the contrast of the tomogram is not inverted
    # the tomogram is centered at the tomogram center
    for pos in BLOBS / scale + np.array(shape) // 2:
        assert_allclose(_local_center_of_mass(tomo, pos), pos, atol=0.15)


@pytest.mark.parametrize("output_spacing", [5.0, 6.4, 8.0, 10.0, 13.7])
@pytest.mark.parametrize("patch_size", [16, 64])
def test_even_crop_margin(output_spacing, patch_size):
    from acryo.loader._pseudo import _even_crop_margin

    margin = _even_crop_margin(patch_size, patch_size // 4, output_spacing, 5.0)
    assert abs(margin - patch_size // 4) <= 4
    padded = 2 * (patch_size + 2 * margin)
    assert round(padded * output_spacing / 5.0) % 2 == 0


def test_reconstruct_tomogram_invalid_input(blob_tilt_series):
    loader = _loader(blob_tilt_series, Molecules(BLOBS + CENTER))
    with pytest.raises(ValueError):
        loader.reconstruct_tomogram((10, 10, 10), patch_size=0)
    with pytest.raises(ValueError):
        loader.reconstruct_tomogram((10, 10, 10), patch_size=15)
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
    sigma = 1.5 * PIXEL_SPACING / 10 / scale
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
    """Copy of the tilt series with randomly shifted tilt images (Å)."""
    out = copy.copy(ts)
    rng = np.random.default_rng(seed)
    shifts = rng.uniform(-amplitude, amplitude, size=ts.sample_translations.shape)
    out.sample_translations = ts.sample_translations + torch.as_tensor(shifts).float()
    return out


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
    loader = _loader(_perturbed(ts, 12.0), mole, scale=scale, tomo_hand=tomo_hand)
    refined = loader.refine_tilt_series([(mole, template)], max_shift=2.0)
    error = refined.tilt_series.sample_translations - ts.sample_translations
    assert error.abs().max() < 1.0  # Angstrom
    assert refined.tilt_series.local_shifts_2d is None


def test_refine_local_shifts(tmp_path):
    centers, rot = _random_particles(60, seed=2)
    rng = np.random.default_rng(3)
    coefs = torch.as_tensor(rng.normal(scale=4.0, size=(TILT_ANGLES.size, 2, 3)))
    half_extent = torch.tensor(IMAGE_SHAPE) * PIXEL_SPACING / 2

    def field(projected):
        # bilinear deformation of the tilt images, up to ~10 Å
        y, x = (projected[..., -2:] / half_extent).unbind(-1)
        terms = torch.stack([y, x, y * x], dim=-1)[..., None, :]  # (n, T, 1, 3)
        return (coefs * terms).sum(-1).float()

    ts = _make_tilt_series(
        tmp_path / "ts.mrc", _particle_points(centers, rot), local_shifts_2d=field
    )
    initial = copy.copy(ts)
    initial.local_shifts_2d = None
    mole = Molecules(centers + CENTER, rot)
    loader = _loader(initial, mole)
    refined = loader.refine_tilt_series(
        [(mole, _particle_template(0.5, 16))], grid_shape=(2, 2)
    )
    points = torch.as_tensor(centers * 10).float()
    true_pos = ts.project_points(points)
    error_before = (initial.project_points(points) - true_pos).norm(dim=-1)
    error_after = (refined.tilt_series.project_points(points) - true_pos).norm(dim=-1)
    assert error_before.mean() > 2.5
    assert error_after.mean() < 0.5
    # local shifts are also applied to the (z, y, x, w) output
    zyxw = refined.tilt_series.project_points(points, output_zyxw=True)
    assert_allclose(
        zyxw[..., 1:3], refined.tilt_series.project_points(points), atol=1e-3
    )
    # the refined tilt series can be used for reconstruction
    assert refined.reconstruct_tomogram((16, 40, 40), patch_size=16).shape == (
        16,
        40,
        40,
    )


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
    loader = _loader(_perturbed(ts, 12.0), mole0)
    refined = loader.refine_tilt_series(
        [
            (mole0, _particle_template(0.5, 16)),
            (mole1, _particle_template(0.5, 16, other)),
        ]
    )
    error = refined.tilt_series.sample_translations - ts.sample_translations
    assert error.abs().max() < 1.0


def test_refined_loader(tmp_path):
    centers, rot = _random_particles(20, seed=6)
    ts = _make_tilt_series(tmp_path / "ts.mrc", _particle_points(centers, rot))
    mole = Molecules(centers + CENTER, rot)
    initial = _perturbed(ts, 15.0)
    translations = initial.sample_translations.clone()
    loader = _loader(initial, mole, output_shape=(15, 15, 15))
    template = _particle_template(0.5, 15)
    images = loader._images.get(1)[1].copy()
    refined = loader.refine_tilt_series([(mole, template)])
    assert_allclose(loader._images.get(1)[1], images)  # the cache is not modified
    assert isinstance(refined, PseudoSubtomogramLoader)
    assert refined.tilt_series is not initial
    assert loader.tilt_series is initial
    assert_allclose(initial.sample_translations, translations)  # not modified
    assert refined.molecules is loader.molecules
    assert refined._images.get(1)[1] is loader._images.get(1)[1]  # shared images

    # subtomograms are sharper after refinement
    def _score(ldr: PseudoSubtomogramLoader) -> float:
        avg = ldr.average()
        return np.corrcoef(avg.ravel(), template.ravel())[0, 1]

    assert _score(refined) > _score(loader) + 0.05


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
    from torch_tilt_series import load_tilt_series_images, preprocess_tilt_series_images
    from acryo.loader._pseudo import _TiltSeriesImages

    ref = preprocess_tilt_series_images(
        load_tilt_series_images(blob_tilt_series), **kwargs
    )
    _, images = _TiltSeriesImages(blob_tilt_series, True, kwargs).get(1)
    assert_allclose(images, ref.numpy(), atol=1e-4)
    # cached in memory, and read-only because the cache is shared
    assert type(images) is np.ndarray
    assert images.dtype == np.float32
    assert not images.flags.writeable


@pytest.mark.parametrize("dtype", [np.float32, np.int16, np.uint16, np.float16])
@pytest.mark.parametrize("extended_header", [False, True])
def test_mrc_images_are_read_one_by_one(tmp_path, monkeypatch, dtype, extended_header):
    from acryo.loader._pseudo import _MrcImageStack

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
    stack = _MrcImageStack(path)
    assert stack.shape == data.shape
    images = list(stack.iter_images([3, 0, -1]))
    for image, index in zip(images, [3, 0, -1]):
        assert image.dtype == np.float32
        assert_allclose(image, data[index].astype(np.float32))
    with pytest.raises(IndexError):
        list(stack.iter_images([5]))


@pytest.mark.parametrize("shape", [(64, 80), (63, 81), (66, 70)])
@pytest.mark.parametrize("binsize", [2, 3])
def test_fourier_bin_keeps_center(shape, binsize):
    from acryo.loader._pseudo import _fourier_bin

    shift = np.array([6.0, -9.0])  # from the center, in the original pixels
    pos = np.array(shape) // 2 + shift
    yy, xx = np.indices(shape, dtype=np.float32)
    image = np.exp(-((yy - pos[0]) ** 2 + (xx - pos[1]) ** 2) / (2 * 3.0**2))
    binned = _fourier_bin(torch.as_tensor(image), binsize).numpy()
    expected = np.array(binned.shape) // 2 + shift / binsize
    weight = np.clip(binned, 0, None) ** 2
    com = (np.indices(binned.shape) * weight).sum(axis=(1, 2)) / weight.sum()
    assert_allclose(com, expected, atol=0.05)


def test_tilt_images_are_loaded_once_per_binsize(blob_tilt_series, monkeypatch):
    from acryo.loader._pseudo import _TiltSeriesImages

    loaded: list[int] = []
    _load = _TiltSeriesImages._load

    def _recorded_load(self, binsize):
        loaded.append(binsize)
        return _load(self, binsize)

    monkeypatch.setattr(_TiltSeriesImages, "_load", _recorded_load)
    loader = _loader(blob_tilt_series, Molecules(BLOBS + CENTER))
    assert loaded == []  # lazy
    loader.average()
    loader.head(3).average()
    loader.binning(2).replace(output_shape=(7, 7, 7)).average()
    loader.binning(2).average(output_shape=(7, 7, 7))
    loader.replace(scale=0.7).average()  # binsize 1 is used again
    assert loaded == [1, 2]


def test_image_indices(tmp_path, blob_tilt_series):
    # the same tilt images with an excluded image at the beginning
    with mrcfile.open(blob_tilt_series.image_path) as mrc:
        data = mrc.data.copy()
    path = tmp_path / "with_excluded.mrc"
    with mrcfile.new(path) as mrc:
        mrc.set_data(np.concatenate([np.ones_like(data[:1]), data]))
    ts = copy.copy(blob_tilt_series)
    ts.image_path = path
    ts.image_indices = torch.arange(1, data.shape[0] + 1)
    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=9))
    ref = _loader(blob_tilt_series, mole).asnumpy()
    assert_allclose(_loader(ts, mole).asnumpy(), ref, atol=1e-5)


@pytest.mark.parametrize("scale", [1.0, 1.5])  # binsize 2 and 3
def test_binned_tilt_images(blob_tilt_series, monkeypatch, scale):
    from acryo.loader._pseudo import _TiltSeriesImages

    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=10))
    loader = _loader(blob_tilt_series, mole, scale=scale, output_shape=(9, 9, 9))
    assert loader._images.binsize_for(scale * 10) == round(scale * 2)
    binned = loader.asnumpy()
    monkeypatch.setattr(_TiltSeriesImages, "binsize_for", lambda self, spacing: 1)
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
