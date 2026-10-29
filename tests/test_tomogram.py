import dataclasses

import numpy as np
from numpy.testing import assert_allclose
import pytest

from acryo.projection import AffineShifts2D, TiltSeries, reconstruct_tomogram
from acryo.projection import _io

IMAGE_SHAPE = (96, 112)
IMAGE_SCALE = 0.5  # nm
SIGMA = 1.5  # pixel
# (z, y, x) in nm relative to the tomogram center
BLOBS = np.array(
    [[0.0, 0.0, 0.0], [3.0, -12.0, 14.0], [-4.0, 10.0, -13.0], [2.0, 11.0, 12.0]]
)


def _tilt_series_and_images(points: np.ndarray = BLOBS, mags=None):
    """Tilt series and its images of Gaussian blobs.

    If given, each image is magnified by the (2, 2) matrix of ``mags``.
    """
    rng = np.random.default_rng(0)
    angles = np.arange(-60.0, 61.0, 3.0)
    ts = TiltSeries(
        tilt_angles=angles,
        tilt_axis_angle=84.0,
        sample_translations=rng.normal(scale=1.0, size=(angles.size, 2)),
        local_shifts_2d=None if mags is None else AffineShifts2D(mags),
        scale=IMAGE_SCALE,
    )
    projected = ts.project_points(points) / IMAGE_SCALE + np.array(IMAGE_SHAPE) // 2
    if mags is None:
        mags = np.tile(np.eye(2), (angles.size, 1, 1))
    inverse = np.linalg.inv(mags).astype(np.float32)
    yy, xx = np.indices(IMAGE_SHAPE, dtype=np.float32)
    images = np.zeros((angles.size,) + IMAGE_SHAPE, dtype=np.float32)
    for proj in projected:
        for t, (py, px) in enumerate(proj):
            # the blobs are also magnified
            (a, b), (c, d) = inverse[t]
            dy, dx = yy - py, xx - px
            r2 = (a * dy + b * dx) ** 2 + (c * dy + d * dx) ** 2
            images[t] += np.exp(-r2 / (2 * SIGMA**2))
    return ts, images


def _local_center_of_mass(tomo: np.ndarray, pos: np.ndarray, r: int = 4):
    corner = np.round(pos).astype(int) - r
    sl = tuple(slice(c, c + 2 * r + 1) for c in corner)
    sub = np.clip(tomo[sl].astype(np.float64), 0, None) ** 4
    inds = np.indices(sub.shape).reshape(3, -1)
    return (inds * sub.ravel()).sum(axis=1) / sub.sum() + corner


@pytest.mark.parametrize("scale", [0.5, 0.8])
@pytest.mark.parametrize("shape", [(24, 60, 70), (25, 61, 67)])
@pytest.mark.parametrize("patch_size, blend_margin", [(16, None), (15, 3), (20, 2)])
def test_blobs_are_reconstructed(scale, shape, patch_size, blend_margin):
    ts, images = _tilt_series_and_images()
    tomo = reconstruct_tomogram(
        ts,
        images,
        shape,
        scale=scale,
        patch_size=patch_size,
        blend_margin=blend_margin,
    )
    assert tomo.shape == shape
    assert tomo.dtype == np.float16  # default
    for pos in BLOBS / scale + np.array(shape) // 2:
        # blobs are elongated along z by the missing wedge
        error = np.abs(_local_center_of_mass(tomo, pos) - pos)
        assert np.all(error < [0.25, 0.1, 0.1])


def test_patches_are_blended_seamlessly():
    ts, images = _tilt_series_and_images()
    shape = (24, 60, 70)
    # a single patch covers the whole tomogram
    ref = reconstruct_tomogram(ts, images, shape, patch_size=80, blend_margin=0)
    for patch_size in [16, 24]:
        tomo = reconstruct_tomogram(ts, images, shape, patch_size=patch_size)
        assert np.corrcoef(tomo.ravel(), ref.ravel())[0, 1] > 0.95


def test_magnified_tilt_images():
    # The magnification of each tilt image is corrected inside the patches, so the
    # tomogram is the same as that of the images without magnification.
    rng = np.random.default_rng(1)
    mags = 1.03 * np.eye(2) + rng.uniform(-0.03, 0.03, size=(41, 2, 2))
    shape = (24, 60, 70)
    ts_ref, images_ref = _tilt_series_and_images()
    ts, images = _tilt_series_and_images(mags=mags)
    for patch_size in [16, 80]:
        kwargs = dict(patch_size=patch_size, dtype=np.float32)
        ref = reconstruct_tomogram(ts_ref, images_ref, shape, **kwargs)
        tomo = reconstruct_tomogram(ts, images, shape, **kwargs)
        assert np.corrcoef(tomo.ravel(), ref.ravel())[0, 1] > 0.99
        for pos in BLOBS / IMAGE_SCALE + np.array(shape) // 2:
            error = np.abs(_local_center_of_mass(tomo, pos) - pos)
            assert np.all(error < [0.25, 0.1, 0.1])


def test_default_scale_and_workers():
    ts, images = _tilt_series_and_images()
    shape = (20, 40, 50)
    ref = reconstruct_tomogram(
        ts, images, shape, scale=IMAGE_SCALE, patch_size=16, dtype=np.float32
    )
    tomo = reconstruct_tomogram(
        ts, images, shape, patch_size=16, num_workers=1, dtype=np.float32
    )
    assert_allclose(tomo, ref, atol=1e-5)


def test_dtype():
    ts, images = _tilt_series_and_images()
    shape = (20, 40, 50)
    ref = reconstruct_tomogram(ts, images, shape, patch_size=16, dtype=np.float32)
    assert ref.dtype == np.float32
    tomo = reconstruct_tomogram(ts, images, shape, patch_size=16)
    assert tomo.dtype == np.float16
    assert_allclose(tomo, ref, rtol=2e-3, atol=1e-4)
    with pytest.raises(ValueError):
        reconstruct_tomogram(ts, images, shape, patch_size=16, dtype=np.float64)
    with pytest.raises(ValueError):
        reconstruct_tomogram(ts, images, shape, patch_size=16, dtype=np.int16)


def test_float16_overflow():
    ts, images = _tilt_series_and_images()
    with pytest.warns(RuntimeWarning, match="float16"):
        tomo = reconstruct_tomogram(ts, images * 1e7, (10, 20, 20), patch_size=16)
    assert np.isinf(tomo).any()


def test_written_slab_by_slab(monkeypatch):
    slabs: list[tuple[int, int]] = []
    _write = _io.VolumeWriter.write

    def _record(self, start, slab):
        slabs.append((start, slab.shape[0]))
        return _write(self, start, slab)

    monkeypatch.setattr(_io.VolumeWriter, "write", _record)
    ts, images = _tilt_series_and_images()
    shape = (50, 30, 30)
    reconstruct_tomogram(ts, images, shape, patch_size=8, blend_margin=2)
    # contiguous slabs, each of which is not larger than a patch
    assert [start for start, _ in slabs] == list(
        np.cumsum([0] + [depth for _, depth in slabs[:-1]])
    )
    assert sum(depth for _, depth in slabs) == shape[0]
    assert max(depth for _, depth in slabs) <= 8 + 2 * 2


def test_save_path(tmp_path):
    mrcfile = pytest.importorskip("mrcfile")
    ts, images = _tilt_series_and_images()
    shape = (21, 40, 50)
    kwargs = dict(scale=0.8, patch_size=16, dtype=np.float32)
    ref = reconstruct_tomogram(ts, images, shape, **kwargs)
    path = tmp_path / "tomo.mrc"
    out = reconstruct_tomogram(ts, images, shape, save_path=path, **kwargs)
    assert isinstance(out, np.memmap)
    assert not out.flags.writeable
    assert_allclose(out, ref, atol=1e-5)
    del out
    with mrcfile.open(path) as mrc:
        assert mrc.data.shape == shape
        assert mrc.header.mode == 2
        assert_allclose(mrc.data, ref, atol=1e-5)
        assert mrc.voxel_size.x == pytest.approx(8.0)
        header = mrc.header
        assert header.dmin == pytest.approx(ref.min(), abs=1e-5)
        assert header.dmax == pytest.approx(ref.max(), abs=1e-5)
        assert header.dmean == pytest.approx(ref.mean(), abs=1e-5)
        assert header.rms == pytest.approx(ref.std(), rel=1e-4)
    # float16 by default
    path16 = tmp_path / "tomo16.mrc"
    out = reconstruct_tomogram(
        ts, images, shape, scale=0.8, patch_size=16, save_path=path16
    )
    assert out.dtype == np.float16
    del out
    with mrcfile.open(path16) as mrc:
        assert mrc.header.mode == 12
        assert mrc.data.dtype == np.float16
        assert_allclose(mrc.data, ref, rtol=2e-3, atol=1e-4)
        # statistics of the stored values
        stored = mrc.data.astype(np.float64)
        assert mrc.header.dmax == pytest.approx(stored.max(), abs=1e-6)
        assert mrc.header.dmean == pytest.approx(stored.mean(), abs=1e-6)
    # existing file is not overwritten by default
    with pytest.raises(ValueError):
        reconstruct_tomogram(ts, images, shape, patch_size=16, save_path=path)
    out = reconstruct_tomogram(
        ts, images, (10, 20, 20), patch_size=16, save_path=path, overwrite=True
    )
    assert out.shape == (10, 20, 20)
    del out


def test_file_is_closed_on_error(tmp_path, monkeypatch):
    pytest.importorskip("mrcfile")
    from acryo.projection import DirectFourierReconstructor

    def _fail(self, positions, rotations=None):
        raise RuntimeError("failed")

    monkeypatch.setattr(DirectFourierReconstructor, "__call__", _fail)
    ts, images = _tilt_series_and_images()
    path = tmp_path / "tomo.mrc"
    with pytest.raises(RuntimeError):
        reconstruct_tomogram(ts, images, (10, 20, 20), patch_size=8, save_path=path)
    path.unlink()  # the file is closed


def test_local_alignment_of_patches(monkeypatch):
    from acryo.projection import DirectFourierReconstructor, LocalAlignment

    ts, images = _tilt_series_and_images()
    rng = np.random.default_rng(1)
    local = LocalAlignment(
        origin=[-10.0, -10.0],
        spacing=[20.0, 20.0],
        tilt_angles=rng.normal(scale=2.0, size=(2, 2, ts.ntilts)),
        mode="full",
    )
    calls = []
    _call = DirectFourierReconstructor.__call__

    def _call_record(self, positions, rotations=None):
        calls.append(rotations)
        return _call(self, positions, rotations)

    monkeypatch.setattr(DirectFourierReconstructor, "__call__", _call_record)
    reconstruct_tomogram(ts, images, (10, 20, 20), patch_size=10)
    assert len(calls) > 0 and all(rot is None for rot in calls)
    calls.clear()
    # only shifted in the "shift" mode
    shifted = ts.replace(local_shifts_2d=dataclasses.replace(local, mode="shift"))
    reconstruct_tomogram(shifted, images, (10, 20, 20), patch_size=10)
    assert len(calls) > 0 and all(rot is None for rot in calls)
    calls.clear()
    # each patch is reconstructed by the projection around its center
    ts = ts.replace(local_shifts_2d=local)
    reconstruct_tomogram(ts, images, (10, 20, 20), patch_size=10)
    assert len(calls) == 4
    rotations = np.stack(calls)
    assert rotations.shape == (4, ts.ntilts, 3, 3)
    assert np.abs(rotations - rotations[0]).max() > 1e-3


def test_invalid_input():
    ts, images = _tilt_series_and_images()
    with pytest.raises(ValueError):
        reconstruct_tomogram(ts.replace(scale=None), images, (10, 10, 10))
    with pytest.raises(ValueError):
        reconstruct_tomogram(ts, images[1:], (10, 10, 10))
    with pytest.raises(ValueError):
        reconstruct_tomogram(ts, images, (10, 10))
    with pytest.raises(ValueError):
        reconstruct_tomogram(ts, images, (10, 0, 10))
    with pytest.raises(ValueError):
        reconstruct_tomogram(ts, images, (10, 10, 10), patch_size=0)
    with pytest.raises(ValueError):
        reconstruct_tomogram(ts, images, (10, 10, 10), blend_margin=-1)
    with pytest.raises(ValueError):
        reconstruct_tomogram(ts, images, (10, 10, 10), num_workers=0)
