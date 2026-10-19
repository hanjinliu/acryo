import dataclasses

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from acryo.ctf import CTFModel, PatchCTFCorrection, TiltSeriesCTF, _ImodDefocus
from acryo.projection import TiltImagePreprocessor, TiltSeries


def _ctf(ntilts: int, **kwargs) -> TiltSeriesCTF:
    return TiltSeriesCTF(defocus_u=np.linspace(-2.0, -3.0, ntilts), **kwargs)


def test_tilt_series_ctf():
    ctf = _ctf(4, defocus_v=-2.5, astigmatism_angle=[0, 10, 20, 30], voltage=200)
    assert ctf.ntilts == 4
    assert_allclose(ctf.defocus_v, -2.5)
    assert_allclose(ctf.astigmatism_angle, [0, 10, 20, 30])
    assert ctf.wave_length == pytest.approx(CTFModel.from_kv(200, 2.7).wave_length / 10)
    with pytest.raises(ValueError):
        ctf.defocus_u[0] = 0.0  # read-only
    selected = ctf.select_tilts([3, 1])
    assert_allclose(selected.defocus_u, ctf.defocus_u[[3, 1]])
    assert_allclose(selected.astigmatism_angle, [30, 10])
    assert selected.voltage == 200
    out = TiltSeriesCTF.from_dict(ctf.to_dict())
    for name in ["defocus_u", "defocus_v", "astigmatism_angle"]:
        assert_allclose(getattr(out, name), getattr(ctf, name))
    assert out.voltage == ctf.voltage
    repr(ctf)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"defocus_u": []},
        {"defocus_u": [[-1.0]]},
        {"defocus_u": [-1.0, -2.0], "defocus_v": [-1.0, -2.0, -3.0]},
        {"defocus_u": [-1.0], "voltage": 0},
        {"defocus_u": [-1.0], "spherical_aberration": -1},
        {"defocus_u": [-1.0], "amplitude_contrast": 1.5},
    ],
)
def test_tilt_series_ctf_invalid_input(kwargs):
    with pytest.raises(ValueError):
        TiltSeriesCTF(**kwargs)


def test_tilt_series_with_ctf():
    ctf = _ctf(3)
    ts = TiltSeries(tilt_angles=[-30, 0, 30], tilt_axis_angle=80.0, ctf=ctf)
    selected = ts.select_tilts([2, 0])
    assert_allclose(selected.ctf.defocus_u, ctf.defocus_u[[2, 0]])
    out = TiltSeries.from_json(ts.dump_json())
    assert_allclose(out.ctf.defocus_u, ctf.defocus_u)
    with pytest.raises(ValueError):
        TiltSeries(tilt_angles=[-30, 0], tilt_axis_angle=80.0, ctf=ctf)
    with pytest.raises(TypeError):
        TiltSeries(tilt_angles=[0], tilt_axis_angle=80.0, ctf=-1.0)  # type: ignore
    with pytest.raises(ValueError):
        TiltSeries(tilt_angles=[0], tilt_axis_angle=80.0).ctf_correction()


@pytest.mark.parametrize("hand", [-1, 1])
def test_defocus_gradient_imod_convention(hand):
    # tilt axis is vertical (along y) on the detector
    angles = np.array([-40.0, 0.0, 40.0])
    ts = TiltSeries(tilt_angles=angles, tilt_axis_angle=0.0, ctf=_ctf(3))
    correction = ts.ctf_correction(hand=hand)
    assert correction.hand == hand
    for t, angle in enumerate(angles):
        left = correction.defocus(t, (0.0, -100.0))
        right = correction.defocus(t, (0.0, 100.0))
        assert correction.defocus(t, (50.0, 0.0))[0] == pytest.approx(
            ts.ctf.defocus_u[t]
        )
        # The depth on the right is -100 * tan(angle) nm, so the underfocus changes by
        # -hand * 100 * tan(angle) nm, and the defocus (μm) by the opposite.
        expected = hand * 100 * np.tan(np.radians(angle)) / 1000
        assert left[0] - ts.ctf.defocus_u[t] == pytest.approx(-expected)
        assert right[0] - ts.ctf.defocus_u[t] == pytest.approx(expected)
        if angle > 0 and hand == -1:
            # the right side is more underfocused (negative defocus)
            assert right[0] < left[0]


def test_defocus_at_projected_points():
    """The defocus is that of the points on the plane z = 0 of the tomogram."""
    rng = np.random.default_rng(3)
    ntilts = 5
    levelled2tomo = np.eye(4)
    levelled2tomo[:3, :3] = Rotation.from_rotvec([0.05, -0.03, 0.4]).as_matrix()
    levelled2tomo[:3, 3] = [3.0, -5.0, 8.0]
    ts = TiltSeries(
        tilt_angles=np.linspace(-50, 50, ntilts),
        tilt_axis_angle=rng.normal(85, 2, ntilts),
        sample_translations=rng.normal(scale=10, size=(ntilts, 2)),
        x_tilts=rng.normal(scale=3, size=ntilts),
        levelled2tomo=levelled2tomo,
        ctf=_ctf(ntilts, defocus_v=-2.2),
    )
    correction = ts.ctf_correction(hand=1)
    points = np.zeros((20, 3))
    points[:, 1:] = rng.uniform(-200, 200, (20, 2))  # on the plane z = 0
    mtx = ts.projection_matrices @ ts.tomo2sample
    homogeneous = np.concatenate([points, np.ones((20, 1))], axis=1)
    projected = np.einsum("tij,nj->nti", mtx, homogeneous)  # (N, T, 4)
    center_depth = mtx[:, 0, 3]
    for t in range(ntilts):
        for proj in projected[:, t]:
            # RELION: the underfocus (positive) increases by hand * depth
            depth = proj[0] - center_depth[t]
            du, dv = correction.defocus(t, (proj[1], proj[2]))
            assert du == pytest.approx(ts.ctf.defocus_u[t] - depth / 1000)
            assert dv == pytest.approx(ts.ctf.defocus_v[t] - depth / 1000)


def test_patch_ctf_correction_select_tilts():
    ts = TiltSeries(tilt_angles=[-30, 0, 30, 60], tilt_axis_angle=85.0, ctf=_ctf(4))
    correction = ts.ctf_correction(hand=1)
    selected = correction.select_tilts([3, 1])
    assert isinstance(selected, PatchCTFCorrection)
    assert selected.hand == 1
    assert selected.defocus(0, (10.0, 20.0)) == correction.defocus(3, (10.0, 20.0))
    assert selected.defocus(1, (10.0, 20.0)) == correction.defocus(1, (10.0, 20.0))
    freq = np.linspace(0, 2, 50)[np.newaxis]
    with pytest.raises(IndexError):
        selected(freq, np.zeros((1, 1)), 2)
    with pytest.raises(ValueError):
        PatchCTFCorrection(ts.ctf, np.zeros((3, 2)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        PatchCTFCorrection(ts.ctf, np.zeros((4, 2)), np.zeros((4, 2)), hand=0)
    # the preprocessor converts it by `select_tilts`
    preprocessor = TiltImagePreprocessor().with_filters(correction)
    (out,) = preprocessor.select_tilts([3, 1]).filters
    assert isinstance(out, PatchCTFCorrection)
    assert_allclose(out.ctf.defocus_u, ts.ctf.defocus_u[[3, 1]])


def _uniform_correction(**kwargs) -> PatchCTFCorrection:
    """The CTF correction of a tilt without the defocus gradient."""
    ctf = TiltSeriesCTF(**kwargs)
    return PatchCTFCorrection(ctf, np.zeros((1, 2)), np.zeros((1, 2)))


def test_patch_ctf_same_as_ctf_model():
    model = CTFModel.from_kv(300, 2.7, defocus=-2.5, amplitude=0.1)
    correction = _uniform_correction(
        defocus_u=[-2.5], astigmatism_angle=17.0, amplitude_contrast=0.1
    )
    freq = np.linspace(0, 3.0, 500)
    sign = correction(freq[:, np.newaxis], np.zeros((1, 1)), 0, position=(5.0, 3.0))
    assert sign.dtype == np.float32
    assert np.mean(sign[:, 0] == np.sign(model.simulate(freq))) > 0.99


@pytest.mark.parametrize("direction", [0.0, 25.0, 70.0, 120.0])
def test_patch_ctf_astigmatism(direction):
    model = CTFModel.from_kv(300, 2.7)
    defocus_u, defocus_v, angle = -2.0, -2.6, 30.0
    correction = _uniform_correction(
        defocus_u=[defocus_u], defocus_v=[defocus_v], astigmatism_angle=angle
    )
    rad = np.radians(direction)
    freq = np.linspace(0, 2.5, 500)
    sign = correction(freq * np.sin(rad), freq * np.cos(rad), 0)
    # defocus along the direction (counterclockwise from x to y)
    cos = np.cos(2 * np.radians(direction - angle))
    defocus = (defocus_u + defocus_v) / 2 + (defocus_u - defocus_v) / 2 * cos
    expected = np.sign(dataclasses.replace(model, defocus=defocus).simulate(freq))
    assert np.mean(sign == expected) > 0.99


def test_patch_ctf_correction_restores_contrast():
    """CTF of a tilted image is corrected better patch by patch."""
    scale, shape, angle = 0.25, (256, 512), 60.0
    rng = np.random.default_rng(0)
    points = np.zeros(shape, dtype=np.float32)
    points[tuple(rng.integers(0, s, 400) for s in shape)] = 1.0
    truth = ndi.gaussian_filter(points, 1.0)
    model = CTFModel.from_kv(300, 2.7, defocus=-2.0)
    # the right side is more underfocused at positive tilt angles (hand = -1)
    xs = (np.arange(shape[1]) - shape[1] // 2) * scale
    observed = np.empty(shape, dtype=np.float32)
    for x0 in range(0, shape[1], 8):
        defocus = -2.0 - np.tan(np.radians(angle)) * xs[x0 + 4] / 1000
        strip = dataclasses.replace(model, defocus=defocus).apply_ctf(truth, scale)
        observed[:, x0 : x0 + 8] = strip[:, x0 : x0 + 8]
    base = TiltImagePreprocessor(
        subtract_background=False, padding=32, patch_grid=(1, 8), overlap=16
    )

    def _corr(hand: int | None) -> float:
        if hand is None:
            preprocessor = base.with_ctf(model)  # the same defocus for all patches
        else:
            ctf = TiltSeriesCTF(defocus_u=[-2.0])
            ts = TiltSeries(tilt_angles=[angle], tilt_axis_angle=0.0, ctf=ctf)
            preprocessor = base.with_filters(ts.ctf_correction(hand=hand))
        out = preprocessor(observed, scale=scale)
        out, ref = out - ndi.gaussian_filter(out, 4), truth - ndi.gaussian_filter(
            truth, 4
        )
        return np.corrcoef(out.ravel(), ref.ravel())[0, 1]

    corr_patch, corr_whole, corr_wrong = _corr(-1), _corr(None), _corr(1)
    assert corr_patch > corr_whole + 0.01
    assert corr_whole > corr_wrong + 0.01


def test_imod_defocus_version2(tmp_path):
    path = tmp_path / "TS.defocus"
    path.write_text("1 1 -60.0 -60.0 4000 2\n2 2 -57.0 -57.0 4100\n4 6 -51 -45 4400\n")
    defocus = _ImodDefocus.read(path)
    assert_allclose(defocus.start_view, [1, 2, 4])
    assert_allclose(defocus.defocus1, [4000, 4100, 4400])
    assert_allclose(defocus.defocus2, defocus.defocus1)
    # interpolated between the middles of the ranges, and the nearest outside
    d1, d2, angle = defocus.at_views([1, 2, 3, 5, 7])
    assert_allclose(d1, [4000, 4100, 4200, 4400, 4400])
    assert_allclose(d2, d1)
    assert_allclose(angle, 0)


def test_imod_defocus_version1_numbered_from_zero(tmp_path):
    path = tmp_path / "TS.defocus"
    path.write_text("0 0 -60.0 -60.0 4000\n1 1 -57.0 -57.0 4100\n")
    defocus = _ImodDefocus.read(path)
    assert_allclose(defocus.start_view, [1, 2])
    assert_allclose(defocus.at_views([1, 2])[0], [4000, 4100])


@pytest.mark.parametrize("flags, angle_factor", [(1, 1.0), (3, np.pi / 180)])
def test_imod_defocus_version3_astigmatism(tmp_path, flags, angle_factor):
    path = tmp_path / "TS.defocus"
    lines = [f"{flags} 0 0. 0. 0 3"]
    lines += [
        f"{v} {v} 0. 0. {4000 + v} {3800 + v} {(10 * v) * angle_factor}"
        for v in [1, 2, 3]
    ]
    path.write_text("\n".join(lines) + "\n")
    defocus = _ImodDefocus.read(path)
    d1, d2, angle = defocus.at_views([1, 3])
    assert_allclose(d1, [4001, 4003])
    assert_allclose(d2, [3801, 3803])
    assert_allclose(angle, [10, 30])
    assert_allclose(defocus.phase_shift, 0)


def test_imod_defocus_version3_phase_shift(tmp_path):
    path = tmp_path / "TS.defocus"
    # astigmatism (1), phase shift (4) in radians (8) and cut-on (32)
    path.write_text("45 0 0. 0. 0 3\n1 1 -50 -50 3665 3472 -62.1 1.5 0.17\n")
    defocus = _ImodDefocus.read(path)
    assert_allclose(defocus.defocus2, [3472])
    assert_allclose(defocus.phase_shift, [np.rad2deg(1.5)])


@pytest.mark.parametrize("text", ["", "1 0 0. 0. 0 3\n1 1 -50 -50 3665\n"])
def test_imod_defocus_invalid(tmp_path, text):
    path = tmp_path / "TS.defocus"
    path.write_text(text)
    with pytest.raises(ValueError):
        _ImodDefocus.read(path)
