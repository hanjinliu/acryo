import dataclasses
import json

import numpy as np
from numpy.testing import assert_allclose
import pytest

from acryo.projection import LocalShifts2D, TiltSeries

NTILTS = 21


def _random_tilt_series(seed: int = 0, **kwargs) -> TiltSeries:
    rng = np.random.default_rng(seed)
    sample2levelled = np.eye(4)
    sample2levelled[:3, :3] = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    sample2levelled[:3, 3] = rng.normal(scale=3.0, size=3)
    levelled2tomo = np.eye(4)
    levelled2tomo[:3, 3] = rng.normal(scale=3.0, size=3)
    params = dict(
        tilt_angles=np.linspace(-60, 60, NTILTS),
        tilt_axis_angle=rng.normal(85, 1, NTILTS),
        sample_translations=rng.normal(scale=3.0, size=(NTILTS, 2)),
        x_tilts=rng.normal(size=NTILTS),
        sample2levelled=sample2levelled,
        levelled2tomo=levelled2tomo,
        local_shifts_2d=LocalShifts2D(
            rng.normal(scale=2.0, size=(NTILTS, 2, 3, 4)), [300.0, 400.0]
        ),
        image_path="tilt_series.mrc",
        image_indices=np.arange(NTILTS) + 1,
        scale=0.25,
    )
    params.update(kwargs)
    return TiltSeries(**params)


def test_project_points_simple_geometry():
    ts = TiltSeries(
        tilt_angles=[0.0, 90.0, 0.0],
        tilt_axis_angle=[0.0, 0.0, 90.0],
        sample_translations=[[0.0, 0.0], [0.0, 0.0], [1.0, -2.0]],
    )
    point = np.array([3.0, 4.0, 5.0])  # (z, y, x)
    projected = ts.project_points(point)
    assert projected.shape == (3, 2)
    assert_allclose(projected[0], [4.0, 5.0], atol=1e-12)  # no tilt
    assert_allclose(projected[1], [4.0, 3.0], atol=1e-12)  # x' = z at 90 degrees
    # the tilt axis is rotated by 90 degrees, and then translated
    assert_allclose(projected[2], [5.0 + 1.0, -4.0 - 2.0], atol=1e-12)
    points = np.zeros((4, 5, 3))
    assert ts.project_points(points).shape == (4, 5, 3, 2)


def test_detector2tomo():
    ts = _random_tilt_series()
    rot = ts.detector2tomo
    assert rot.shape == (NTILTS, 3, 3)
    assert_allclose(
        rot @ rot.transpose(0, 2, 1), np.tile(np.eye(3), (NTILTS, 1, 1)), atol=1e-12
    )
    # the (y, x) axes of the detector are mapped back to the detector
    linear = ts.projection_matrices[:, 1:3, :3] @ ts.tomo2sample[:3, :3]
    assert_allclose(
        linear @ rot[:, :, 1:], np.tile(np.eye(2), (NTILTS, 1, 1)), atol=1e-12
    )


def test_local_shifts():
    ts = _random_tilt_series(local_shifts_2d=None)
    points = np.random.default_rng(1).uniform(-100, 100, size=(10, 3))
    ref = ts.project_points(points)
    shifted = ts.replace(
        local_shifts=lambda p: np.full_like(p, 0.0),
        local_shifts_2d=lambda p: np.full_like(p, 1.5),
    )
    assert_allclose(shifted.project_points(points), ref + 1.5)
    # local shifts in the sample space are applied before projection
    shift_3d = np.array([0.0, 2.0, -1.0])
    moved = ts.replace(local_shifts=lambda p: np.broadcast_to(shift_3d, p.shape))
    sample = (np.c_[points, np.ones(10)] @ ts.tomo2sample.T)[:, :3] + shift_3d
    expected = ts.replace(sample2levelled=None, levelled2tomo=None).project_points(
        sample
    )
    assert_allclose(moved.project_points(points), expected, atol=1e-10)


def test_local_shifts_2d_interpolation():
    rng = np.random.default_rng(2)
    a, b, c = rng.normal(size=(3, NTILTS, 2))
    half_extent = np.array([100.0, 200.0])
    # bilinear field a * y + b * x + c is exactly interpolated
    yy, xx = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 5), indexing="ij")
    grid = a[..., None, None] * yy + b[..., None, None] * xx + c[..., None, None]
    shifts = LocalShifts2D(grid, half_extent)
    projected = rng.uniform(-1, 1, size=(30, NTILTS, 2)) * half_extent
    normed = projected / half_extent
    expected = a * normed[..., 0:1] + b * normed[..., 1:2] + c
    assert_allclose(shifts(projected), expected, atol=1e-12)
    # constant outside the images
    outside = projected * 3
    clipped = np.clip(outside / half_extent, -1, 1)
    expected = a * clipped[..., 0:1] + b * clipped[..., 1:2] + c
    assert_allclose(shifts(outside), expected, atol=1e-12)
    # a single control point is a uniform shift, and the base is added
    uniform = LocalShifts2D(c[..., None, None], half_extent, base=shifts)
    assert_allclose(uniform(projected), shifts(projected) + c, atol=1e-12)


def test_immutable_and_replace():
    ts = _random_tilt_series()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ts.scale = 1.0  # type: ignore
    with pytest.raises(ValueError):
        ts.sample_translations[0, 0] = 1.0
    other = ts.replace(scale=1.0)
    assert other.scale == 1.0 and ts.scale == 0.25
    assert other.ntilts == NTILTS
    assert_allclose(other.tilt_angles, ts.tilt_angles)
    repr(ts)


def test_json_roundtrip():
    ts = _random_tilt_series()
    out = TiltSeries.from_json(ts.dump_json())
    for name in [
        "tilt_angles",
        "tilt_axis_angle",
        "sample_translations",
        "x_tilts",
        "sample2levelled",
        "levelled2tomo",
        "image_indices",
    ]:
        assert_allclose(getattr(out, name), getattr(ts, name), err_msg=name)
    assert out.image_path == ts.image_path
    assert out.scale == ts.scale
    points = np.random.default_rng(1).normal(scale=50.0, size=(10, 3))
    assert_allclose(out.project_points(points), ts.project_points(points))


def test_json_without_geometry():
    # json files dumped by older versions
    ts = TiltSeries(tilt_angles=[-30.0, 0.0, 30.0], tilt_axis_angle=85.0, scale=0.5)
    data = json.loads(ts.dump_json())
    for key in ["x_tilts", "sample2levelled", "levelled2tomo"]:
        data.pop(key)
    out = TiltSeries.from_json(json.dumps(data))
    assert_allclose(out.x_tilts, [0.0, 0.0, 0.0])
    assert_allclose(out.sample2tomo, np.eye(4))


def test_broadcast_and_defaults():
    ts = TiltSeries(tilt_angles=[-30.0, 0.0, 30.0], tilt_axis_angle=85.0)
    assert_allclose(ts.tilt_axis_angle, [85.0, 85.0, 85.0])
    assert_allclose(ts.x_tilts, [0.0, 0.0, 0.0])
    assert_allclose(ts.sample_translations, np.zeros((3, 2)))
    assert_allclose(ts.sample2tomo, np.eye(4))
    assert ts.image_path is None and ts.image_indices is None and ts.scale is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tilt_angles": []},
        {"tilt_angles": np.zeros((2, 3))},
        {"tilt_axis_angle": [1.0, 2.0]},
        {"sample_translations": np.zeros((3, 3))},
        {"x_tilts": [0.0]},
        {"sample2levelled": np.eye(3)},
        {"image_indices": [0, 1]},
        {"image_indices": [0.0, 1.0, 2.0]},
        {"scale": 0.0},
    ],
)
def test_invalid_input(kwargs):
    params = {"tilt_angles": [-30.0, 0.0, 30.0], "tilt_axis_angle": 85.0}
    params.update(kwargs)
    with pytest.raises(ValueError):
        TiltSeries(**params)


def test_from_aretomo_output(tmp_path, monkeypatch):
    alnfile = pytest.importorskip("alnfile")
    pd = pytest.importorskip("pandas")

    rng = np.random.default_rng(7)
    n = 5
    df = pd.DataFrame(
        {
            "sec": [1, 2, 4, 5, 6],
            "rot": rng.normal(85, 1, n),
            "tilt": np.linspace(-40, 40, n),
            "tx": rng.normal(scale=5, size=n),
            "ty": rng.normal(scale=5, size=n),
        }
    )
    monkeypatch.setattr(alnfile, "read", lambda path: df.copy())
    path = tmp_path / "TS_01.aln"
    ts = TiltSeries.from_aretomo_output(path, scale=0.2)
    assert ts.image_path == tmp_path / "TS_01.mrc"
    assert ts.scale == pytest.approx(0.2)
    assert_allclose(ts.image_indices, [0, 1, 3, 4, 5])  # 1-based sections
    assert_allclose(ts.tilt_angles, df["tilt"])
    assert_allclose(ts.tilt_axis_angle, df["rot"])
    # (y, x) shifts in pixels to nm
    assert_allclose(ts.sample_translations, df[["ty", "tx"]].to_numpy() * 0.2)
