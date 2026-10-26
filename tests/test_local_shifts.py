import dataclasses
import json
import warnings
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from acryo.projection import (
    AffineShifts2D,
    FunctionShifts,
    LocalAlignment,
    LocalShiftModel,
    LocalShifts2D,
    TiltSeries,
)

NTILTS = 9


def _tilt_series(**kwargs) -> TiltSeries:
    rng = np.random.default_rng(0)
    params = dict(
        tilt_angles=np.linspace(-60, 60, NTILTS),
        tilt_axis_angle=rng.normal(85, 1, NTILTS),
        sample_translations=rng.normal(scale=3.0, size=(NTILTS, 2)),
        x_tilts=rng.normal(size=NTILTS),
    )
    params.update(kwargs)
    return TiltSeries(**params)


def _uniform_alignment(grid_shape=(3, 4), **kwargs) -> LocalAlignment:
    """Local alignment of the same parameters in all the local areas."""
    shape = tuple(grid_shape) + (NTILTS,)
    params = {"tilt_angles": np.zeros(NTILTS)}
    params.update(kwargs)
    tail = {
        "tilt_angles": (),
        "x_tilts": (),
        "z_factors": (2,),
        "matrices": (2, 2),
        "translations": (2,),
    }
    arrays = {
        k: np.broadcast_to(np.asarray(v, dtype=np.float64), shape + tail[k])
        for k, v in params.items()
    }
    return LocalAlignment(origin=[-100.0, -150.0], spacing=[100.0, 100.0], **arrays)


def _rotate(ts: TiltSeries, yx: np.ndarray) -> np.ndarray:
    """Rotate (N, T, 2) vectors by the tilt axis angles, as scope2detector."""
    rad = np.deg2rad(ts.tilt_axis_angle)
    c, s = np.cos(rad), np.sin(rad)
    return np.stack(
        [c * yx[..., 0] + s * yx[..., 1], -s * yx[..., 0] + c * yx[..., 1]], axis=-1
    )


def _points(seed=1, n=20) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-300, 300, size=(n, 3)) * [0.2, 1, 1]


def test_local_alignment_identity():
    ts = _tilt_series()
    points = _points()
    shifted = ts.replace(local_shifts_2d=_uniform_alignment())
    assert_allclose(shifted.project_points(points), ts.project_points(points))


def test_local_alignment_angles():
    ts = _tilt_series()
    points = _points()
    rng = np.random.default_rng(2)
    d_tilt, d_xtilt = rng.normal(size=(2, NTILTS))
    local = _uniform_alignment(tilt_angles=d_tilt, x_tilts=d_xtilt)
    expected = ts.replace(
        tilt_angles=ts.tilt_angles + d_tilt, x_tilts=ts.x_tilts + d_xtilt
    ).project_points(points)
    out = ts.replace(local_shifts_2d=local).project_points(points)
    assert_allclose(out, expected, atol=1e-10)


def test_local_alignment_refinement():
    ts = _tilt_series()
    points = _points()
    rng = np.random.default_rng(3)
    matrices = np.eye(2) + rng.normal(scale=0.01, size=(NTILTS, 2, 2))
    translations = rng.normal(size=(NTILTS, 2))
    z_factors = rng.normal(scale=0.01, size=(NTILTS, 2))
    local = _uniform_alignment(
        matrices=matrices, translations=translations, z_factors=z_factors
    )
    out = ts.replace(local_shifts_2d=local).project_points(points)
    # projection before the rotation by the tilt axis angle
    scope = np.einsum("tij,nj->nti", ts.sample2scope[:, 1:3, :3], points)
    refined = np.einsum("tij,ntj->nti", matrices, scope) + translations
    refined += z_factors * points[:, 0, None, None]
    expected = ts.project_points(points) + _rotate(ts, refined - scope)
    assert_allclose(out, expected, atol=1e-10)


def test_local_alignment_interpolation():
    ts = _tilt_series()
    rng = np.random.default_rng(4)
    gy, gx = 3, 4
    translations = rng.normal(size=(gy, gx, NTILTS, 2))
    local = LocalAlignment(
        origin=[-100.0, -150.0],
        spacing=[100.0, 100.0],
        tilt_angles=np.zeros((gy, gx, NTILTS)),
        translations=translations,
    )
    shifted = ts.replace(local_shifts_2d=local)
    yy, xx = np.meshgrid(
        -100.0 + 100 * np.arange(gy), -150.0 + 100 * np.arange(gx), indexing="ij"
    )
    centers = np.stack([yy, xx], axis=-1)  # (gy, gx, 2)
    z = rng.uniform(-30, 30, size=(gy, gx, 1))
    points = np.concatenate([z, centers], axis=-1).reshape(-1, 3)
    out = shifted.project_points(points) - ts.project_points(points)
    assert_allclose(out, _rotate(ts, translations.reshape(-1, NTILTS, 2)), atol=1e-10)
    # bilinear between the local areas, and clamped outside
    point = np.array([[0.0, -50.0, -125.0], [0.0, 500.0, -500.0]])
    out = shifted.project_points(point) - ts.project_points(point)
    expected = [
        np.tensordot([0.75, 0.25], translations[0:2, 0:2].mean(axis=0), axes=1),
        translations[-1, 0],
    ]
    assert_allclose(out, _rotate(ts, np.stack(expected)), atol=1e-10)


def test_affine_shifts():
    ts = _tilt_series()
    points = _points()
    rng = np.random.default_rng(5)
    matrices = np.eye(2) + rng.normal(scale=0.01, size=(NTILTS, 2, 2))
    translations = rng.normal(size=(NTILTS, 2))
    affine = AffineShifts2D(matrices, translations)
    projected = ts.project_points(points)
    out = ts.replace(local_shifts_2d=affine).project_points(points)
    expected = np.einsum("tij,ntj->nti", matrices, projected) + translations
    assert_allclose(out, expected, atol=1e-10)


def test_base_and_select_tilts():
    ts = _tilt_series()
    points = _points()
    rng = np.random.default_rng(6)
    affine = AffineShifts2D(
        np.eye(2) + rng.normal(scale=0.01, size=(NTILTS, 2, 2)),
        rng.normal(size=(NTILTS, 2)),
    )
    local = _uniform_alignment(tilt_angles=rng.normal(size=NTILTS), grid_shape=(2, 2))
    chained = LocalShifts2D(
        rng.normal(size=(NTILTS, 2, 2, 3)),
        [300.0, 300.0],
        base=dataclasses.replace(local, base=affine),
    )
    full = ts.replace(local_shifts_2d=chained).project_points(points)
    # the base is applied first
    expected = ts.project_points(points)
    for model in [
        affine,
        local,
        LocalShifts2D(chained.control_points, chained.half_extent),
    ]:
        expected = expected + model.shifts(ts, points, expected)
    assert_allclose(full, expected, atol=1e-10)

    idx = [0, 3, 4, 8]
    sub = ts.replace(
        tilt_angles=ts.tilt_angles[idx],
        tilt_axis_angle=ts.tilt_axis_angle[idx],
        sample_translations=ts.sample_translations[idx],
        x_tilts=ts.x_tilts[idx],
        local_shifts_2d=chained.select_tilts(idx),
    )
    assert_allclose(sub.project_points(points), full[:, idx], atol=1e-10)


def test_linear_part():
    rng = np.random.default_rng(10)
    m_outer, m_inner = np.eye(2) + rng.normal(scale=0.03, size=(2, NTILTS, 2, 2))
    inner = AffineShifts2D(m_inner, rng.normal(size=(NTILTS, 2)))
    outer = AffineShifts2D(m_outer, rng.normal(size=(NTILTS, 2)), base=inner)
    expected = m_outer @ m_inner
    assert_allclose(outer.linear_part(), expected)
    assert_allclose(outer.select_tilts([0, 2]).linear_part(), expected[[0, 2]])
    # the position-dependent models keep the linear part of the base
    smooth = LocalShifts2D(rng.normal(size=(NTILTS, 2, 2, 3)), [300.0, 300.0])
    assert smooth.linear_part() is None
    assert_allclose(dataclasses.replace(smooth, base=outer).linear_part(), expected)
    local = _uniform_alignment(tilt_angles=rng.normal(size=NTILTS), grid_shape=(2, 2))
    assert local.linear_part() is None
    assert_allclose(dataclasses.replace(local, base=inner).linear_part(), m_inner)
    assert FunctionShifts(lambda p: np.zeros_like(p)).linear_part() is None


def test_local_detector2tomo():
    ts = _tilt_series()
    assert_allclose(ts.local_detector2tomo, ts.detector2tomo)
    smooth = LocalShifts2D(np.ones((NTILTS, 2, 2, 3)), [300.0, 300.0])
    assert_allclose(
        ts.replace(local_shifts_2d=smooth).local_detector2tomo, ts.detector2tomo
    )
    rng = np.random.default_rng(11)
    m_outer, m_inner = np.eye(2) + rng.normal(scale=0.03, size=(2, NTILTS, 2, 2))
    affine = AffineShifts2D(
        m_outer,
        rng.normal(size=(NTILTS, 2)),
        base=AffineShifts2D(m_inner, rng.normal(size=(NTILTS, 2))),
    )
    ts = ts.replace(local_shifts_2d=affine)
    # the offsets from a point are mapped to the offsets from its projection
    point = _points(n=1)[0]
    offsets = rng.normal(scale=20.0, size=(5, 3))
    projected = ts.project_points(point + offsets) - ts.project_points(point)
    inverse = np.linalg.inv(ts.local_detector2tomo)
    expected = np.einsum("tij,nj->nti", inverse, offsets)[..., 1:]
    assert_allclose(projected, expected, atol=1e-8)


def test_position_dependent():
    ts = _tilt_series()
    assert not ts.position_dependent
    rng = np.random.default_rng(12)
    affine = AffineShifts2D(np.eye(2) + rng.normal(scale=0.03, size=(NTILTS, 2, 2)))
    assert not ts.replace(local_shifts_2d=affine).position_dependent
    local = _uniform_alignment(tilt_angles=rng.normal(size=NTILTS))
    smooth = LocalShifts2D(np.ones((NTILTS, 2, 2, 3)), [300.0, 300.0])
    for model in [
        local,
        smooth,
        AffineShifts2D(affine.matrices, base=local),
        FunctionShifts(lambda p: np.zeros_like(p)),
    ]:
        assert model.position_dependent
        assert ts.replace(local_shifts_2d=model).position_dependent
    assert ts.replace(local_shifts=lambda p: np.zeros_like(p)).position_dependent


def test_local_detector2tomo_at():
    rng = np.random.default_rng(13)
    points = _points(n=6)
    affine = AffineShifts2D(np.eye(2) + rng.normal(scale=0.03, size=(NTILTS, 2, 2)))
    ts = _tilt_series(local_shifts_2d=affine)
    out = ts.local_detector2tomo_at(points)
    assert out.shape == (6, NTILTS, 3, 3)
    assert_allclose(out, np.broadcast_to(ts.local_detector2tomo, out.shape), atol=1e-8)
    nested = ts.local_detector2tomo_at(points.reshape(2, 3, 3))
    assert nested.shape == (2, 3, NTILTS, 3, 3)

    # the local alignments of the areas differ in the tilt angles and the rotation
    gy, gx = 3, 4
    angles = np.deg2rad(rng.normal(scale=1.0, size=(gy, gx, NTILTS)))
    c, s = np.cos(angles), np.sin(angles)
    local = LocalAlignment(
        origin=[-100.0, -150.0],
        spacing=[100.0, 100.0],
        tilt_angles=rng.normal(scale=2.0, size=(gy, gx, NTILTS)),
        x_tilts=rng.normal(scale=1.0, size=(gy, gx, NTILTS)),
        matrices=np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2),
    )
    ts = ts.replace(local_shifts_2d=AffineShifts2D(affine.matrices, base=local))
    maps = ts.local_detector2tomo_at(points)
    # the small offsets from each point are mapped to the offsets from its projection
    offsets = rng.normal(scale=0.2, size=(6, 3))
    projected = ts.project_points(points + offsets) - ts.project_points(points)
    inverse = np.linalg.inv(maps)[:, :, 1:, :]
    approx = np.einsum("ntij,nj->nti", inverse, offsets)
    error = np.abs(approx - projected).max()
    assert error < 1e-4
    # not by the linear maps of the whole tilt series
    inverse = np.linalg.inv(ts.local_detector2tomo)[:, 1:, :]
    error_global = np.abs(np.einsum("tij,nj->nti", inverse, offsets) - projected).max()
    assert error_global > 10 * error
    with pytest.raises(ValueError):
        ts.local_detector2tomo_at(points[:, :2])


def test_serialization():
    ts = _tilt_series()
    points = _points()
    rng = np.random.default_rng(7)
    local = LocalAlignment(
        origin=[-100.0, -150.0],
        spacing=[100.0, 120.0],
        tilt_angles=rng.normal(size=(2, 3, NTILTS)),
        x_tilts=rng.normal(size=(2, 3, NTILTS)),
        z_factors=rng.normal(scale=0.01, size=(2, 3, NTILTS, 2)),
        matrices=np.eye(2) + rng.normal(scale=0.01, size=(2, 3, NTILTS, 2, 2)),
        translations=rng.normal(size=(2, 3, NTILTS, 2)),
        base=AffineShifts2D(np.eye(2) + rng.normal(scale=0.01, size=(NTILTS, 2, 2))),
    )
    model = LocalShifts2D(rng.normal(size=(NTILTS, 2, 2, 2)), [200, 300], base=local)
    ts = ts.replace(local_shifts_2d=model)
    data = json.loads(json.dumps(model.to_dict()))
    assert isinstance(LocalShiftModel.from_dict(data).base, LocalAlignment)
    out = TiltSeries.from_json(ts.dump_json())
    assert isinstance(out.local_shifts_2d, LocalShifts2D)
    assert_allclose(out.project_points(points), ts.project_points(points))
    # default parameters are omitted
    plain = _uniform_alignment().to_dict()
    assert set(plain) == {"type", "origin", "spacing", "tilt_angles", "base"}
    with pytest.raises(ValueError):
        LocalShiftModel.from_dict({"type": "Unknown"})


def test_functions_are_not_serializable():
    ts = _tilt_series(local_shifts_2d=lambda p: np.zeros_like(p))
    assert isinstance(ts.local_shifts_2d, FunctionShifts)
    with pytest.raises(TypeError):
        ts.dump_json()
    with pytest.raises(TypeError):
        _tilt_series(local_shifts=lambda p: np.zeros_like(p)).dump_json()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tilt_angles": np.zeros((2, NTILTS))},
        {"x_tilts": np.zeros((2, 3, NTILTS))},
        {"matrices": np.zeros((2, 2, NTILTS, 2))},
        {"spacing": [0.0, 1.0]},
        {"origin": [0.0]},
    ],
)
def test_invalid_local_alignment(kwargs):
    params = dict(
        origin=[0.0, 0.0], spacing=[1.0, 1.0], tilt_angles=np.zeros((2, 2, NTILTS))
    )
    params.update(kwargs)
    with pytest.raises(ValueError):
        LocalAlignment(**params)


def test_local_alignment_needs_points():
    local = _uniform_alignment()
    with pytest.raises(ValueError):
        LocalShifts2D(np.zeros((NTILTS, 2, 1, 1)), [1.0, 1.0], base=local)(
            np.zeros((3, NTILTS, 2))
        )
    ts = TiltSeries(
        tilt_angles=np.zeros(NTILTS + 1), tilt_axis_angle=0.0, local_shifts_2d=local
    )
    with pytest.raises(ValueError):
        ts.project_points(np.zeros(3))


# --- IMOD ---------------------------------------------------------------------------


def _write_imod_local(path: Path, params: dict, pixel_angstrom: float):
    """Write a local alignment file of IMOD with X tilts and Z factors."""
    ny, nx, ntilts = params["dtilt"].shape
    x0, y0, dx, dy = params["grid"]
    lines = [f"{nx} {ny} {x0} {y0} {dx} {dy} 1 {pixel_angstrom} 1"]
    for iy in range(ny):
        for ix in range(nx):
            lines.append(" ".join(f"{v:.12g}" for v in params["dtilt"][iy, ix]))
            lines.append(" ".join(f"{v:.12g}" for v in params["dxtilt"][iy, ix]))
            lines.append(" ".join(f"{v:.12g}" for v in params["zf"][iy, ix].ravel()))
            for v in range(ntilts):
                lines.append(" ".join(f"{a:.12g}" for a in params["xf"][iy, ix, v]))
    path.write_text("\n".join(lines) + "\n")


def _imod_projection(points, df, xaxistilt, local, scale, shape, z_factors):
    """Projection by the IMOD model in pixels of (x, y), written independently."""
    ny_img, nx_img = shape
    out = np.empty((len(points), len(df), 2))
    for n, (z, y, x) in enumerate(np.asarray(points) / scale):
        big_x, big_y = x + nx_img / 2, y + ny_img / 2  # specimen coordinates
        weights = []
        if local is not None:
            x0, y0, dx, dy = local["grid"]
            gy, gx = local["dtilt"].shape[:2]
            fx = np.clip((big_x - x0) / dx, 0, gx - 1)
            fy = np.clip((big_y - y0) / dy, 0, gy - 1)
            ix0, iy0 = min(int(fx), gx - 2), min(int(fy), gy - 2)
            tx, ty = fx - ix0, fy - iy0
            weights = [
                ((iy0, ix0), (1 - tx) * (1 - ty)),
                ((iy0, ix0 + 1), tx * (1 - ty)),
                ((iy0 + 1, ix0), (1 - tx) * ty),
                ((iy0 + 1, ix0 + 1), tx * ty),
            ]
        for t, (_, row) in enumerate(df.iterrows()):
            v = row["idx_tilt"]
            areas = weights or [(None, 1.0)]
            aligned = np.zeros(2)
            for area, w in areas:
                tilt, xtilt = row["tlt"], -xaxistilt + row["xtilt"]
                a, d, zf = np.eye(2), np.zeros(2), np.zeros(2)
                if area is not None:
                    tilt += local["dtilt"][area][v]
                    xtilt += local["dxtilt"][area][v]
                    xf = local["xf"][area][v]
                    a, d = xf[:4].reshape(2, 2), xf[4:]
                    if z_factors:
                        zf = local["zf"][area][v]
                al, th = np.deg2rad(xtilt), np.deg2rad(tilt)
                y1 = np.cos(al) * y - np.sin(al) * z
                z1 = np.sin(al) * y + np.cos(al) * z
                p = np.array([np.cos(th) * x + np.sin(th) * z1, y1])
                aligned += w * (a @ p + d + zf * z)
            m = np.array(
                [[row["xf_a11"], row["xf_a12"]], [row["xf_a21"], row["xf_a22"]]]
            )
            shift = np.array([row["xf_dx"], row["xf_dy"]])
            out[n, t] = np.linalg.solve(m, aligned - shift)
    return out[..., ::-1] * scale  # (y, x) in nm


def _etomo_dataframe(rng, n=7, magnification=True):
    pd = pytest.importorskip("pandas")
    angles = np.radians(rng.normal(-177, 1, n))
    mag = rng.uniform(0.99, 1.01, n) if magnification else np.ones(n)
    stretch = rng.normal(scale=0.003, size=n) if magnification else np.zeros(n)
    excluded = np.zeros(n, dtype=bool)
    excluded[2] = True
    return pd.DataFrame(
        {
            "image_path": [f"TS_01.st[{i}]" for i in range(n)],
            "idx_tilt": np.arange(n),
            "tilt_axis_angle": [177.0] * n,  # nominal value in .edf
            "rawtlt": np.linspace(-45, 45, n),
            "tlt": np.linspace(-45, 45, n) + 0.3,
            "xtilt": rng.normal(size=n),
            "xaxistilt": [1.5] * n,
            "xf_a11": mag * np.cos(angles) + stretch,
            "xf_a12": -mag * np.sin(angles),
            "xf_a21": mag * np.sin(angles),
            "xf_a22": mag * np.cos(angles),
            "xf_dx": rng.normal(scale=5, size=n),
            "xf_dy": rng.normal(scale=5, size=n),
            "excluded": excluded,
        }
    )


@pytest.mark.parametrize("magnification", [True, False])
def test_from_etomo_directory_global(tmp_path, monkeypatch, magnification):
    etomofiles = pytest.importorskip("etomofiles")
    df = _etomo_dataframe(np.random.default_rng(9), magnification=magnification)
    monkeypatch.setattr(etomofiles, "read", lambda path: df.copy())
    scale = 0.2
    ts = TiltSeries.from_etomo_directory(tmp_path, scale=scale, ctf=False)
    kept = df.loc[~df["excluded"]]
    assert ts.ntilts == 6
    assert_allclose(ts.image_indices, [0, 1, 3, 4, 5, 6])
    points = _points(10)
    expected = _imod_projection(points, kept, 1.5, None, scale, (100, 100), False)
    assert_allclose(ts.project_points(points), expected, atol=1e-8)
    if magnification:
        assert isinstance(ts.local_shifts_2d, AffineShifts2D)
    else:
        assert ts.local_shifts_2d is None
        angles = np.degrees(np.arctan2(kept["xf_a21"], kept["xf_a11"]))
        assert_allclose(np.cos(np.radians(ts.tilt_axis_angle + angles)), 1.0)


def test_from_etomo_directory_local(tmp_path, monkeypatch):
    etomofiles = pytest.importorskip("etomofiles")
    rng = np.random.default_rng(11)
    df = _etomo_dataframe(rng)
    monkeypatch.setattr(etomofiles, "read", lambda path: df.copy())
    scale, shape = 0.25, (1200, 1000)  # (NY, NX) of the images
    gy, gx, n = 3, 4, len(df)
    xf = np.zeros((gy, gx, n, 6))
    xf[..., :4] = np.eye(2).ravel() + rng.normal(scale=0.005, size=(gy, gx, n, 4))
    xf[..., 4:] = rng.normal(scale=2, size=(gy, gx, n, 2))
    local = {
        "grid": (150.0, 200.0, 250.0, 300.0),  # X_start, Y_start, dX, dY in pixels
        "dtilt": rng.normal(scale=0.5, size=(gy, gx, n)),
        "dxtilt": rng.normal(scale=0.5, size=(gy, gx, n)),
        "zf": rng.normal(scale=0.01, size=(gy, gx, n, 2)),
        "xf": xf,
    }
    _write_imod_local(tmp_path / "TS_01local.xf", local, scale * 10)
    tilt_com = ["# comment", "$tilt -StandardInput", "LOCALFILE TS_01local.xf"]
    tilt_com.append(f"FULLIMAGE {shape[1]} {shape[0]}")
    (tmp_path / "tilt.com").write_text("\n".join(tilt_com) + "\n")

    kept = df.loc[~df["excluded"]]
    points = np.concatenate(
        [
            rng.uniform(-30, 30, (40, 1)),
            rng.uniform(-0.6, 0.6, (40, 2)) * np.array(shape) * scale,
        ],
        axis=1,
    )
    without = TiltSeries.from_etomo_directory(tmp_path, scale=scale, ctf=False)
    ts = TiltSeries.from_etomo_directory(
        tmp_path, scale=scale, local_alignment=True, ctf=False
    )
    assert isinstance(ts.local_shifts_2d, AffineShifts2D)
    assert isinstance(ts.local_shifts_2d.base, LocalAlignment)
    expected = _imod_projection(points, kept, 1.5, local, scale, shape, False)
    assert_allclose(ts.project_points(points), expected, atol=1e-6)
    assert (
        np.abs(ts.project_points(points) - without.project_points(points)).max() > 0.5
    )

    # the file can be given, and the Z factors are used with ZFACTORFILE
    (tmp_path / "tilt.com").write_text(
        "\n".join(tilt_com[:2] + tilt_com[3:] + ["ZFACTORFILE TS_01.zfac"]) + "\n"
    )
    with pytest.warns(UserWarning, match="ZFACTORFILE"):
        ts = TiltSeries.from_etomo_directory(
            tmp_path, scale=scale, local_alignment="TS_01local.xf", ctf=False
        )
    expected = _imod_projection(points, kept, 1.5, local, scale, shape, True)
    assert_allclose(ts.project_points(points), expected, atol=1e-6)
    out = TiltSeries.from_json(ts.dump_json())
    assert_allclose(out.project_points(points), ts.project_points(points))


def test_from_etomo_directory_no_local_file(tmp_path, monkeypatch):
    etomofiles = pytest.importorskip("etomofiles")
    df = _etomo_dataframe(np.random.default_rng(12))
    monkeypatch.setattr(etomofiles, "read", lambda path: df.copy())
    with pytest.raises(ValueError, match="local alignment"):
        TiltSeries.from_etomo_directory(
            tmp_path, scale=0.2, local_alignment=True, ctf=False
        )


def _etomo_coarse_dataframe(rng, n=7):
    """The DataFrame of etomofiles before the fine alignment (no .xf, .tlt, .xtilt)."""
    df = _etomo_dataframe(rng, n)
    xf_columns = ["xf_a11", "xf_a12", "xf_a21", "xf_a22", "xf_dx", "xf_dy"]
    df[["tlt", "xtilt"] + xf_columns] = np.nan
    df["xaxistilt"] = None
    return df


def _write_imod_xf(path: Path, shifts):
    """Write the (dx, dy) translations as an IMOD transformation file."""
    lines = [f"1 0 0 1 {dx:.7f} {dy:.7f}" for dx, dy in shifts]
    path.write_text("\n".join(lines) + "\n")


@pytest.mark.parametrize("ext", ["prexg", "prexf"])
def test_from_etomo_directory_coarse(tmp_path, monkeypatch, ext):
    etomofiles = pytest.importorskip("etomofiles")
    rng = np.random.default_rng(13)
    df = _etomo_coarse_dataframe(rng)
    monkeypatch.setattr(etomofiles, "read", lambda path: df.copy())
    scale = 0.2
    xg = rng.normal(scale=20, size=(len(df), 2))  # (dx, dy) of the coarse alignment
    if ext == "prexg":
        _write_imod_xf(tmp_path / "TS_01.prexg", xg)
    else:
        # translations to the previous views, from which xftoxg -nfit 0 gives the
        # translations relative to the average
        xf = np.concatenate([np.zeros((1, 2)), np.diff(xg, axis=0)])
        _write_imod_xf(tmp_path / "TS_01.prexf", xf)
        xg = np.cumsum(xf, axis=0)
        xg -= xg.mean(axis=0)
    with pytest.warns(UserWarning, match="coarse alignment"):
        ts = TiltSeries.from_etomo_directory(tmp_path, scale=scale, ctf=False)
    kept = np.flatnonzero(~df["excluded"].to_numpy())
    assert_allclose(ts.image_indices, kept)
    assert_allclose(ts.tilt_angles, df["rawtlt"].to_numpy()[kept])
    assert_allclose(np.cos(np.radians(ts.tilt_axis_angle - 177.0)), 1.0)
    assert ts.local_shifts_2d is None

    # the views of the coarse aligned stack are rotated by -177° to the projections
    points = _points(10)
    th, rot = np.radians(ts.tilt_angles), np.radians(177.0)
    rotation = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    z, y, x = (points / scale).T[..., np.newaxis]
    proj_x = np.cos(th) * x + np.sin(th) * z  # (N, T)
    proj = np.stack([proj_x, np.broadcast_to(y, proj_x.shape)], axis=-1)
    expected = proj @ rotation.T - xg[kept]  # (x, y) in pixels of the raw images
    assert_allclose(ts.project_points(points), expected[..., ::-1] * scale, atol=1e-6)


def test_from_etomo_directory_coarse_errors(tmp_path, monkeypatch):
    etomofiles = pytest.importorskip("etomofiles")
    df = _etomo_coarse_dataframe(np.random.default_rng(14))
    monkeypatch.setattr(etomofiles, "read", lambda path: df.copy())
    with pytest.raises(FileNotFoundError, match="prexg"):
        TiltSeries.from_etomo_directory(tmp_path, scale=0.2, ctf=False)
    _write_imod_xf(tmp_path / "TS_01.prexg", np.zeros((len(df) - 1, 2)))
    with pytest.raises(ValueError, match="views"):
        TiltSeries.from_etomo_directory(tmp_path, scale=0.2, ctf=False)


def _write_ctf_files(path: Path, nviews: int, invert: str | None = None):
    """Write the defocus file (version 3 with astigmatism) and ctfcorrection.com."""
    lines = ["1 0 0. 0. 0 3"]
    for v in range(1, nviews + 1):
        lines.append(f"{v} {v} 0. 0. {3000 + 100 * v} {2900 + 100 * v} {5 * v}")
    (path / "TS_01_ctf.defocus").write_text("\n".join(lines) + "\n")
    com = [
        "$ctfphaseflip -StandardInput",
        "InputStack TS_01.ali",
        "DefocusFile TS_01_ctf.defocus",
        "Voltage 300",
        "SphericalAberration 2.7",
        "AmplitudeContrast 0.07",
    ]
    if invert is not None:
        com.append(f"InvertTiltAngles {invert}".strip())
    (path / "ctfcorrection.com").write_text("\n".join(com) + "\n")


@pytest.mark.parametrize(
    "invert, inverted", [(None, False), ("", True), ("1", True), ("0", False)]
)
def test_from_etomo_directory_ctf(tmp_path, monkeypatch, invert, inverted):
    etomofiles = pytest.importorskip("etomofiles")
    df = _etomo_dataframe(np.random.default_rng(9))
    monkeypatch.setattr(etomofiles, "read", lambda path: df.copy())
    _write_ctf_files(tmp_path, len(df), invert)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        ts = TiltSeries.from_etomo_directory(tmp_path, scale=0.2)
    # the handedness is not read, but warned
    assert any("InvertTiltAngles" in str(w.message) for w in record) == inverted
    views = np.array([1, 2, 4, 5, 6, 7])  # the third view is excluded
    # nm (underfocus positive) to μm (underfocus negative)
    assert_allclose(ts.ctf.defocus_u, -(3000 + 100 * views) / 1000)
    assert_allclose(ts.ctf.defocus_v, -(2900 + 100 * views) / 1000)
    # angles of the raw stack (ctfplotter is not run on the aligned stack)
    assert_allclose(ts.ctf.astigmatism_angle, 5 * views)
    assert ts.ctf.voltage == 300
    assert ts.ctf.spherical_aberration == 2.7
    assert ts.ctf.amplitude_contrast == 0.07
    # as ctfphaseflip, the defocus is at the center of the raw images
    assert ts.ctf.defocus_at == "image_center"
    correction = ts.ctf_correction()
    for t in range(ts.ntilts):
        du, dv = correction.defocus(t, (0.0, 0.0))
        assert du == pytest.approx(ts.ctf.defocus_u[t])
        assert dv == pytest.approx(ts.ctf.defocus_v[t])
    assert TiltSeries.from_etomo_directory(tmp_path, scale=0.2, ctf=False).ctf is None


def test_from_etomo_directory_ctf_of_aligned_stack(tmp_path, monkeypatch):
    etomofiles = pytest.importorskip("etomofiles")
    df = _etomo_dataframe(np.random.default_rng(9))
    monkeypatch.setattr(etomofiles, "read", lambda path: df.copy())
    _write_ctf_files(tmp_path, len(df))
    # the parameters of ctfplotter are used if not in ctfcorrection.com
    (tmp_path / "ctfcorrection.com").unlink()
    com = ["InputStack TS_01.ali", "Voltage 200", "SphericalAberration 2.7"]
    com.append("AmplitudeContrast 0.1")
    (tmp_path / "ctfplotter.com").write_text("\n".join(com) + "\n")
    # the defocus file of the dataset name is used by default
    (tmp_path / "TS_01_ctf.defocus").rename(tmp_path / "TS_01.defocus")
    ts = TiltSeries.from_etomo_directory(tmp_path, scale=0.2)
    views = np.array([1, 2, 4, 5, 6, 7])
    assert ts.ctf.voltage == 200
    # rotated to the angles of the raw stack
    assert_allclose(ts.ctf.astigmatism_angle, 5 * views + ts.tilt_axis_angle)
    # the center of the aligned images is the projection of the tomogram center
    assert ts.ctf.defocus_at == "tomogram_center"


def test_from_etomo_directory_ctf_not_found(tmp_path, monkeypatch):
    etomofiles = pytest.importorskip("etomofiles")
    df = _etomo_dataframe(np.random.default_rng(9))
    monkeypatch.setattr(etomofiles, "read", lambda path: df.copy())
    with pytest.warns(UserWarning, match="Defocus file"):
        ts = TiltSeries.from_etomo_directory(tmp_path, scale=0.2)
    assert ts.ctf is None
    with pytest.raises(FileNotFoundError):
        TiltSeries.from_etomo_directory(tmp_path, scale=0.2, ctf="TS_01.defocus")
    # the defocus file without the microscope parameters
    _write_ctf_files(tmp_path, len(df))
    (tmp_path / "ctfcorrection.com").write_text("DefocusFile TS_01_ctf.defocus\n")
    with pytest.warns(UserWarning, match="VOLTAGE"):
        ts = TiltSeries.from_etomo_directory(tmp_path, scale=0.2)
    assert ts.ctf is None
    with pytest.raises(ValueError, match="VOLTAGE"):
        TiltSeries.from_etomo_directory(tmp_path, scale=0.2, ctf="TS_01_ctf.defocus")
