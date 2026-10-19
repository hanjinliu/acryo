from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from acryo import BackProjectionLoader, Molecules
from acryo.ctf import PatchCTFCorrection
from acryo.projection import TiltSeries

pytest.importorskip("starfile_rs")
mrcfile = pytest.importorskip("mrcfile")

PIXEL_SIZE = 2.5  # Å
NTILTS = 7


def _rotation_xyz(angles, axis: str) -> np.ndarray:
    """Rotation matrices of RELION (xyz, right-handed, in degrees)."""
    rad = np.deg2rad(angles)
    c, s = np.cos(rad), np.sin(rad)
    out = np.tile(np.eye(4), (rad.size, 1, 1))
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    out[:, i, i] = c
    out[:, i, j] = -s
    out[:, j, i] = s
    out[:, j, j] = c
    return out


def _translation(shifts) -> np.ndarray:
    shifts = np.atleast_2d(shifts)
    out = np.tile(np.eye(4), (shifts.shape[0], 1, 1))
    out[:, : shifts.shape[1], 3] = shifts
    return out


def _relion_projection(alignment: dict, image_shape, tomo_shape) -> np.ndarray:
    """Projection matrices of RELION 5 (xyz pixels -> xy pixels of tilt images)."""
    image_center = np.array(image_shape[::-1]) / 2  # (x, y)
    tomo_center = np.array(tomo_shape[::-1]) / 2  # (x, y, z)
    shifts = alignment["shifts_angst"] / PIXEL_SIZE  # (T, 2) in (x, y)
    return (
        _translation(image_center)
        @ _translation(shifts)
        @ _rotation_xyz(alignment["zrot"], "z")
        @ _rotation_xyz(alignment["ytilt"], "y")
        @ _rotation_xyz(alignment["xtilt"], "x")
        @ _translation(-tomo_center)
    )


def _alignment(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "xtilt": rng.normal(scale=2.0, size=NTILTS),
        "ytilt": np.linspace(-45, 45, NTILTS) + rng.normal(scale=0.5, size=NTILTS),
        "zrot": rng.normal(85, 1, NTILTS),
        "shifts_angst": rng.normal(scale=30.0, size=(NTILTS, 2)),
    }


def _write_star(path: Path, blocks: dict[str, dict[str, list]]):
    lines = []
    for name, columns in blocks.items():
        lines += [f"data_{name}", "", "loop_"]
        keys = list(columns)
        lines += [f"_{key} #{i + 1}" for i, key in enumerate(keys)]
        for row in zip(*columns.values()):
            lines.append(" ".join(str(v) for v in row))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def _make_project(
    root: Path,
    image_shape=(40, 50),
    aligned: bool = True,
    tomo_names=("TS_01",),
    seed: int = 0,
    ctf: bool = True,
    hand: int = -1,
) -> tuple[dict, list[np.ndarray]]:
    """Make a RELION project with motion-corrected tilt images and STAR files."""
    rng = np.random.default_rng(seed)
    alignment = _alignment(seed)
    images = []
    for name in tomo_names:
        micrographs = []
        for t in range(NTILTS):
            rel = Path("MotionCorr/job002/frames") / f"{name}_{t:03d}.mrc"
            image = rng.normal(size=image_shape).astype(np.float32)
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            with mrcfile.new(root / rel, overwrite=True) as mrc:
                mrc.set_data(image)
                mrc.voxel_size = PIXEL_SIZE
            micrographs.append(rel.as_posix())
            images.append(image)
        columns = {
            "rlnMicrographName": micrographs,
            "rlnTomoNominalStageTiltAngle": np.linspace(-45, 45, NTILTS).tolist(),
            "rlnTomoNominalTiltAxisAngle": [85.0] * NTILTS,
        }
        if aligned:
            columns["rlnTomoXTilt"] = alignment["xtilt"].tolist()
            columns["rlnTomoYTilt"] = alignment["ytilt"].tolist()
            columns["rlnTomoZRot"] = alignment["zrot"].tolist()
            columns["rlnTomoXShiftAngst"] = alignment["shifts_angst"][:, 0].tolist()
            columns["rlnTomoYShiftAngst"] = alignment["shifts_angst"][:, 1].tolist()
        if ctf:
            for key, values in _defocus().items():
                columns[key] = values.tolist()
        _write_star(
            root / "AlignTiltSeries/job005/tilt_series" / f"{name}.star",
            {name: columns},
        )
    _write_star(
        root / "AlignTiltSeries/job005/tilt_series.star",
        {
            "global": {
                "rlnTomoName": list(tomo_names),
                "rlnTomoTiltSeriesStarFile": [
                    f"AlignTiltSeries/job005/tilt_series/{name}.star"
                    for name in tomo_names
                ],
                "rlnVoltage": [300.0] * len(tomo_names),
                "rlnSphericalAberration": [2.7] * len(tomo_names),
                "rlnAmplitudeContrast": [0.1] * len(tomo_names),
                "rlnTomoHand": [hand] * len(tomo_names),
                "rlnTomoTiltSeriesPixelSize": [PIXEL_SIZE] * len(tomo_names),
            }
        },
    )
    return alignment, images


def _defocus() -> dict[str, np.ndarray]:
    """CTF parameters of the tilts (in Å and degrees, underfocus positive)."""
    defocus_u = 30000.0 + 500 * np.arange(NTILTS)
    return {
        "rlnDefocusU": defocus_u,
        "rlnDefocusV": defocus_u - 300,
        "rlnDefocusAngle": 10.0 * np.arange(NTILTS),
    }


@pytest.mark.parametrize("image_shape", [(40, 50), (41, 53)])
def test_same_projection_as_relion(tmp_path, image_shape):
    alignment, _ = _make_project(tmp_path, image_shape=image_shape)
    ts = TiltSeries.from_relion_tilt_series_star(
        tmp_path / "AlignTiltSeries/job005/tilt_series.star"
    )
    assert ts.ntilts == NTILTS
    assert ts.scale == pytest.approx(PIXEL_SIZE / 10)
    tomo_shape = (31, 64, 70)  # (z, y, x) in pixels
    rng = np.random.default_rng(1)
    voxels = rng.uniform(0, 1, size=(20, 3)) * np.array(tomo_shape)  # (z, y, x)
    matrices = _relion_projection(alignment, image_shape, tomo_shape)
    xyzw = np.concatenate([voxels[:, ::-1], np.ones((20, 1))], axis=1)
    ref = np.einsum("tij,nj->nti", matrices[:, :2], xyzw)[..., ::-1]  # (y, x)
    # centered coordinates in nm
    points = (voxels - np.array(tomo_shape) / 2) * PIXEL_SIZE / 10
    projected = ts.project_points(points) / ts.scale + np.array(image_shape) // 2
    assert_allclose(projected, ref, atol=1e-6)


def test_tilt_images(tmp_path):
    _, images = _make_project(tmp_path)
    ts = TiltSeries.from_relion_tilt_series_star(
        tmp_path / "AlignTiltSeries/job005/tilt_series.star"
    )
    assert isinstance(ts.image_path, tuple)
    assert all(p.is_absolute() and p.exists() for p in ts.image_path)
    assert ts.image_indices is None
    loader = BackProjectionLoader(ts, Molecules.empty(), preprocess=False)
    assert_allclose(loader._images.get(1)[1], np.stack(images))
    repr(ts)


def test_select_tilt_series(tmp_path):
    _make_project(tmp_path, tomo_names=("TS_01", "TS_02", "003"))
    path = tmp_path / "AlignTiltSeries/job005/tilt_series.star"
    with pytest.raises(ValueError, match="tomo_name"):
        TiltSeries.from_relion_tilt_series_star(path)
    with pytest.raises(ValueError, match="not found"):
        TiltSeries.from_relion_tilt_series_star(path, "TS_04")
    for name in ["TS_02", "003"]:
        ts = TiltSeries.from_relion_tilt_series_star(path, name)
        assert ts.image_path[0].name.startswith(f"{name}_")


def test_single_tilt_series_star(tmp_path):
    _make_project(tmp_path)
    path = tmp_path / "AlignTiltSeries/job005/tilt_series/TS_01.star"
    ref = TiltSeries.from_relion_tilt_series_star(
        tmp_path / "AlignTiltSeries/job005/tilt_series.star"
    )
    # the pixel size is read from the header of the tilt image, and the CTF
    # parameters are not read without the global data block
    with pytest.warns(UserWarning, match="global data block"):
        ts = TiltSeries.from_relion_tilt_series_star(path)
    assert ts.ctf is None
    assert ts.scale == pytest.approx(PIXEL_SIZE / 10)
    assert ts.image_path == ref.image_path
    assert_allclose(
        ts.project_points(np.zeros((1, 3))), ref.project_points(np.zeros((1, 3)))
    )
    ts = TiltSeries.from_relion_tilt_series_star(
        path, scale=0.5, project_dir=tmp_path, ctf=False
    )
    assert ts.scale == 0.5


def test_project_dir_not_found(tmp_path):
    _make_project(tmp_path / "project")
    star = tmp_path / "project/AlignTiltSeries/job005/tilt_series/TS_01.star"
    moved = tmp_path / "other/TS_01.star"
    moved.parent.mkdir()
    moved.write_text(star.read_text())
    with pytest.raises(ValueError, match="project_dir"):
        TiltSeries.from_relion_tilt_series_star(moved, ctf=False)
    ts = TiltSeries.from_relion_tilt_series_star(
        moved, project_dir=tmp_path / "project", ctf=False
    )
    assert all(p.exists() for p in ts.image_path)


def test_not_aligned(tmp_path):
    _make_project(tmp_path, aligned=False)
    with pytest.warns(UserWarning, match="not aligned"):
        ts = TiltSeries.from_relion_tilt_series_star(
            tmp_path / "AlignTiltSeries/job005/tilt_series.star"
        )
    assert_allclose(ts.tilt_angles, np.linspace(-45, 45, NTILTS))
    assert_allclose(ts.tilt_axis_angle, 85.0)
    assert_allclose(ts.sample_translations, 0.0)


def test_images_in_stack(tmp_path):
    rng = np.random.default_rng(3)
    stack = rng.normal(size=(NTILTS + 2, 20, 30)).astype(np.float32)
    with mrcfile.new(tmp_path / "stack.mrcs") as mrc:
        mrc.set_data(stack)
        mrc.voxel_size = PIXEL_SIZE
    order = [3, 1, 2, 4, 5, 6, 8]  # 0-based
    _write_star(
        tmp_path / "TS_01.star",
        {
            "TS_01": {
                "rlnMicrographName": [f"{i + 1}@stack.mrcs" for i in order],
                "rlnTomoNominalStageTiltAngle": np.linspace(-45, 45, NTILTS).tolist(),
                "rlnTomoNominalTiltAxisAngle": [85.0] * NTILTS,
            }
        },
    )
    with pytest.warns(UserWarning):
        ts = TiltSeries.from_relion_tilt_series_star(tmp_path / "TS_01.star")
    assert_allclose(ts.image_indices, order)
    loader = BackProjectionLoader(ts, Molecules.empty(), preprocess=False)
    assert_allclose(loader._images.get(1)[1], stack[order])


def test_json_roundtrip(tmp_path):
    _make_project(tmp_path)
    ts = TiltSeries.from_relion_tilt_series_star(
        tmp_path / "AlignTiltSeries/job005/tilt_series.star"
    )
    out = TiltSeries.from_json(ts.dump_json())
    assert out.image_path == ts.image_path
    assert_allclose(out.sample_translations, ts.sample_translations)
    assert_allclose(out.x_tilts, ts.x_tilts)
    assert_allclose(out.ctf.defocus_u, ts.ctf.defocus_u)


def test_ctf(tmp_path):
    _make_project(tmp_path)
    path = tmp_path / "AlignTiltSeries/job005/tilt_series.star"
    ts = TiltSeries.from_relion_tilt_series_star(path)
    defocus = _defocus()
    # Å (underfocus positive) to μm (underfocus negative)
    assert_allclose(ts.ctf.defocus_u, -defocus["rlnDefocusU"] / 1e4)
    assert_allclose(ts.ctf.defocus_v, -defocus["rlnDefocusV"] / 1e4)
    assert_allclose(ts.ctf.astigmatism_angle, defocus["rlnDefocusAngle"])
    assert ts.ctf.voltage == 300.0
    assert ts.ctf.spherical_aberration == 2.7
    assert ts.ctf.amplitude_contrast == 0.1
    assert TiltSeries.from_relion_tilt_series_star(path, ctf=False).ctf is None
    # the CTF is corrected by default
    loader = BackProjectionLoader(ts, Molecules.empty())
    (correction,) = loader.preprocessor.filters
    assert isinstance(correction, PatchCTFCorrection)
    assert correction.hand == -1


def test_ctf_inverted_hand(tmp_path):
    _make_project(tmp_path, hand=1)
    path = tmp_path / "AlignTiltSeries/job005/tilt_series.star"
    # the handedness is not read, but warned
    with pytest.warns(UserWarning, match="rlnTomoHand"):
        ts = TiltSeries.from_relion_tilt_series_star(path)
    assert ts.ctf is not None


def test_ctf_not_estimated(tmp_path):
    _make_project(tmp_path, ctf=False)
    path = tmp_path / "AlignTiltSeries/job005/tilt_series.star"
    with pytest.warns(UserWarning, match="rlnDefocusU"):
        ts = TiltSeries.from_relion_tilt_series_star(path)
    assert ts.ctf is None
    loader = BackProjectionLoader(ts, Molecules.empty())
    assert loader.preprocessor.filters == ()


def test_image_path_length():
    with pytest.raises(ValueError):
        TiltSeries(tilt_angles=[0.0, 1.0], tilt_axis_angle=0.0, image_path=["a.mrc"])
    ts = TiltSeries(tilt_angles=[0.0], tilt_axis_angle=0.0, image_path=["a.mrc"])
    assert ts.image_path == (Path("a.mrc"),)
