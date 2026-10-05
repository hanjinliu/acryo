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


def _make_tilt_series(path, points_nm: np.ndarray, sigma: float = 1.5) -> TiltSeries:
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
    """Record the number of points and the sidelength of each reconstruction."""
    import torch_reconstruct_tomogram.reconstruct as _mod

    calls: list[tuple[int, int]] = []
    _recon = _mod._reconstruct_subvolume

    def _recon_record(tilt_series, images, points, sidelength, **kwargs):
        calls.append((len(points), sidelength))
        return _recon(tilt_series, images, points, sidelength, **kwargs)

    monkeypatch.setattr(_mod, "_reconstruct_subvolume", _recon_record)
    return calls


@pytest.mark.parametrize(
    "batch_size, npoints", [(1, [1] * 5), (2, [2, 2, 1]), (None, [5])]
)
def test_batch_size(blob_tilt_series, recorded_calls, batch_size, npoints):
    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=4))
    ref = _loader(blob_tilt_series, mole, batch_size=len(BLOBS)).asnumpy()
    recorded_calls.clear()
    loader = _loader(blob_tilt_series, mole, batch_size=batch_size)
    stack = loader.construct_dask()
    # subtomograms are reconstructed immediately, and only rotation is delayed
    assert [n for n, _ in recorded_calls] == npoints
    assert_allclose(stack.compute(), ref, rtol=1e-5, atol=1e-5)
    assert [n for n, _ in recorded_calls] == npoints


def test_load_reconstructs_given_subtomograms(blob_tilt_series, recorded_calls):
    mole = Molecules(BLOBS + CENTER, Rotation.random(len(BLOBS), random_state=4))
    loader = _loader(blob_tilt_series, mole)
    loader.load(2)
    loader.load([3, 1])
    loader.load(slice(1, 5, 2))
    assert [n for n, _ in recorded_calls] == [1, 2, 2]


# 4 x 4 x 4 grid of blobs with 4 nm spacing
DENSE_BLOBS = np.stack(
    np.meshgrid(*[np.arange(4) * 4.0 - 6.0] * 3, indexing="ij"), axis=-1
).reshape(-1, 3)


@pytest.mark.parametrize(
    "blobs, chunk_size, calls",
    [
        (DENSE_BLOBS, 15.0, [(1, 30)] * 8),  # 2 x 2 x 2 chunks of 30 pixels
        (DENSE_BLOBS, 1000.0, [(64, 16)]),  # chunks are too large
        (BLOBS, 15.0, [(5, 16)]),  # molecules are sparse
    ],
)
def test_chunk_strategy(tmp_path, recorded_calls, blobs, chunk_size, calls):
    ts = _make_tilt_series(tmp_path / "blobs.mrc", blobs)
    mole = Molecules(blobs + CENTER, Rotation.random(len(blobs), random_state=6))
    loader = _loader(ts, mole, output_shape=(9, 9, 9), chunk_size=chunk_size)
    assert loader.asnumpy().shape == (len(blobs), 9, 9, 9)
    assert recorded_calls == calls


@pytest.mark.parametrize("tomo_hand", [1, -1])
@pytest.mark.parametrize("corner_safe, chunk_size", [(False, 15.0), (True, 20.0)])
def test_chunks_consistent_with_tomogram_reconstruction(
    tmp_path, recorded_calls, corner_safe, chunk_size, tomo_hand
):
    ts = _make_tilt_series(tmp_path / "blobs.mrc", DENSE_BLOBS)
    scale = 0.5
    shape = (60, 100, 100)
    tomo_center = np.array(shape) // 2 * scale
    rot = Rotation.random(len(DENSE_BLOBS), random_state=7)
    mole = Molecules(DENSE_BLOBS + tomo_center, rot)
    kwargs = dict(output_shape=(9, 9, 9), scale=scale, corner_safe=corner_safe)
    loader = PseudoSubtomogramLoader(
        ts,
        mole,
        tomogram_center=tomo_center,
        chunk_size=chunk_size,
        tomo_hand=tomo_hand,
        **kwargs,
    )
    out = loader.asnumpy()
    assert [n for n, _ in recorded_calls] == [1] * 8  # 2 x 2 x 2 chunks
    # NOTE: the contrast of the tomogram is not inverted
    tomo = -loader.reconstruct_tomogram(shape, patch_size=32)
    ref = SubtomogramLoader(tomo, mole, **kwargs).asnumpy()
    for img0, img1 in zip(ref, out):
        assert np.corrcoef(img0.ravel(), img1.ravel())[0, 1] > 0.95


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
    assert binned.chunk_size == loader.chunk_size == 100.0
    assert binned.tomo_hand == loader.tomo_hand == 1
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
        PseudoSubtomogramLoader(np.zeros((4, 4, 4)), mole)
    with pytest.raises(TypeError):
        PseudoSubtomogramLoader(blob_tilt_series, BLOBS)
    with pytest.raises(ValueError):
        PseudoSubtomogramLoader(blob_tilt_series, mole, tomogram_center=(0, 0))
    with pytest.raises(ValueError):
        PseudoSubtomogramLoader(blob_tilt_series, mole, batch_size=0)
    with pytest.raises(ValueError):
        PseudoSubtomogramLoader(blob_tilt_series, mole, chunk_size=0)
    with pytest.raises(ValueError):
        PseudoSubtomogramLoader(blob_tilt_series, mole, tomo_hand=0)
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
