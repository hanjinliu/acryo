import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import fft as sfft
from scipy import ndimage as ndi

from acryo.backend._zncc_2d import ncc_2d, zncc_2d, zncc_landscape_2d
from acryo.ctf import CTFModel, TiltSeriesCTF
from acryo.loader._tilt_refine import (
    ProjectionCTF,
    _bicubic,
    _bilinear_nodes,
    _fourier_resize,
    _lowpass_weight,
    _smoothness_penalty,
    _subpixel_peak,
)
from acryo.projection import LocalShifts2D, TiltSeries


def test_bicubic_gradient():
    rng = np.random.default_rng(0)
    maps = rng.normal(size=(20, 9, 9))
    y, x = rng.uniform(0, 8, size=(2, 20))
    value, grad_y, grad_x = _bicubic(maps, y, x)
    eps = 1e-6
    numerical_y = (_bicubic(maps, y + eps, x)[0] - _bicubic(maps, y - eps, x)[0]) / 2
    numerical_x = (_bicubic(maps, y, x + eps)[0] - _bicubic(maps, y, x - eps)[0]) / 2
    assert_allclose(grad_y, numerical_y / eps, atol=1e-6)
    assert_allclose(grad_x, numerical_x / eps, atol=1e-6)
    # exact at the pixels
    iy, ix = rng.integers(0, 9, size=(2, 20))
    value = _bicubic(maps, iy.astype(np.float64), ix.astype(np.float64))[0]
    assert_allclose(value, maps[np.arange(20), iy, ix], atol=1e-12)


def test_bicubic_properties():
    rng = np.random.default_rng(1)
    y, x = rng.uniform(1, 7, size=(2, 20))
    # constants are reproduced (partition of unity)
    value, grad_y, grad_x = _bicubic(np.full((20, 9, 9), 2.5), y, x)
    assert_allclose(value, 2.5, atol=1e-12)
    assert_allclose(grad_y, 0.0, atol=1e-12)
    assert_allclose(grad_x, 0.0, atol=1e-12)
    # the weights are symmetric, so linear functions are exact at half pixels
    a, b, c = rng.normal(size=3)
    yy, xx = np.indices((9, 9), dtype=np.float64)
    maps = np.broadcast_to(a * yy + b * xx + c, (20, 9, 9))
    yh, xh = np.floor(y) + 0.5, np.floor(x) + 0.5
    assert_allclose(_bicubic(maps, yh, xh)[0], a * yh + b * xh + c, atol=1e-10)


def test_smoothness_penalty_gradient():
    rng = np.random.default_rng(2)
    control = rng.normal(size=(2, 3, 4))
    penalty, grad = _smoothness_penalty(control)
    expected = (np.diff(control, axis=1) ** 2).sum() + (
        np.diff(control, axis=2) ** 2
    ).sum()
    assert penalty == pytest.approx(expected)
    eps = 1e-6
    numerical = np.zeros_like(control)
    for idx in np.ndindex(control.shape):
        plus, minus = control.copy(), control.copy()
        plus[idx] += eps
        minus[idx] -= eps
        diff = _smoothness_penalty(plus)[0] - _smoothness_penalty(minus)[0]
        numerical[idx] = diff / (2 * eps)
    assert_allclose(grad, numerical, atol=1e-5)


@pytest.mark.parametrize("grid_shape", [(2, 2), (3, 4), (1, 3)])
def test_bilinear_nodes_consistent_with_local_shifts(grid_shape):
    rng = np.random.default_rng(3)
    control = rng.normal(size=(1, 2) + grid_shape)
    half_extent = np.array([50.0, 70.0])
    positions = rng.uniform(-1.3, 1.3, size=(30, 2))
    nodes, weights = _bilinear_nodes(positions, grid_shape)
    flat = control[0].reshape(2, -1)
    shifts = (flat[:, nodes] * weights).sum(axis=-1).T
    ref = LocalShifts2D(control, half_extent)(positions[:, None] * half_extent)
    assert_allclose(shifts, ref[:, 0], atol=1e-12)


def test_fourier_resize():
    n, m = 32, 24
    yy, xx = np.indices((n, n)) - n // 2
    sigma = 3.0
    image = np.exp(-(yy**2 + xx**2) / (2 * sigma**2))
    ft = sfft.rfft2(sfft.ifftshift(image))
    resized = sfft.fftshift(sfft.irfft2(_fourier_resize(ft, m), s=(m, m)))
    # the same Gaussian at the pixel size n / m (the sum is kept)
    yy, xx = (np.indices((m, m)) - m // 2) * n / m
    expected = np.exp(-(yy**2 + xx**2) / (2 * sigma**2)) * (n / m) ** 2
    assert_allclose(resized, expected, atol=1e-6)
    # padding and cropping back is the identity
    assert_allclose(_fourier_resize(_fourier_resize(ft, 40), n), ft)


def test_subpixel_peak():
    yy, xx = np.indices((9, 9)) - 4.0
    maps = np.stack(
        [-((yy - dy) ** 2) - (xx - dx) ** 2 for dy, dx in [(0.3, -1.2), (-2.0, 2.4)]]
    )
    assert_allclose(_subpixel_peak(maps), [[0.3, -1.2], [-2.0, 2.4]], atol=1e-10)


def test_ncc_2d():
    rng = np.random.default_rng(4)
    img0 = rng.normal(size=(3, 5, 11, 12)) + 1.0
    img1 = 0.5 * img0 + rng.normal(scale=0.5, size=img0.shape) + 2.0
    zncc = zncc_2d(img0, img1)
    assert zncc.shape == (3, 5)
    for idx in np.ndindex(3, 5):
        expected = np.corrcoef(img0[idx].ravel(), img1[idx].ravel())[0, 1]
        assert zncc[idx] == pytest.approx(expected)
        expected = (img0[idx] * img1[idx]).sum() / np.sqrt(
            (img0[idx] ** 2).sum() * (img1[idx] ** 2).sum()
        )
        assert ncc_2d(img0, img1)[idx] == pytest.approx(expected)
    # invariant to the scale, and to the offset only for ZNCC
    assert_allclose(ncc_2d(img0, 3 * img1), ncc_2d(img0, img1))
    assert_allclose(zncc_2d(img0, 3 * img1 - 5), zncc)
    assert_allclose(zncc_2d(img0, img0), 1.0)
    assert_allclose(zncc_2d(img0, -img0), -1.0)
    # zero images
    assert_allclose(zncc_2d(img0, np.ones_like(img1)), 0.0)


def test_zncc_landscape_2d():
    rng = np.random.default_rng(5)
    size, radius = 24, 4
    img0 = ndi.gaussian_filter(rng.normal(size=(2, size, size)), (0, 1.5, 1.5)) + 1.0
    img1 = rng.normal(size=(2, size, size))
    ft0 = sfft.rfft2(img0)
    ft1 = sfft.rfft2(img1)
    maps = zncc_landscape_2d(ft0, ft1, radius)
    assert maps.shape == (2, 2 * radius + 1, 2 * radius + 1)
    # the center is the ZNCC of the images, and the inputs are not modified
    assert_allclose(maps[:, radius, radius], zncc_2d(img0, img1), atol=1e-6)
    assert_allclose(ft0, sfft.rfft2(img0))
    # the peak is at the shift of the image
    shifted = np.roll(img0, (2, -3), axis=(-2, -1))
    maps = zncc_landscape_2d(sfft.rfft2(shifted), ft0, radius)
    assert_allclose(maps[:, radius + 2, radius - 3], 1.0, atol=1e-6)
    assert np.all(maps.reshape(2, -1).argmax(axis=1) == (radius + 2) * 9 + radius - 3)


def test_lowpass_weight_consistent_with_backend():
    from acryo.backend import Backend

    rng = np.random.default_rng(6)
    image = rng.normal(size=(16, 16)).astype(np.float32)
    expected = Backend("numpy").lowpass_filter(image, cutoff=0.2)
    filtered = sfft.irfft2(sfft.rfft2(image) * _lowpass_weight(16, 0.2), s=(16, 16))
    assert_allclose(filtered, expected, atol=1e-5)


def _ctf_tilt_series() -> TiltSeries:
    ntilts = 3
    ctf = TiltSeriesCTF(np.array([-2.0, -2.5, -3.0]), voltage=200.0)
    return TiltSeries(
        tilt_angles=[-40.0, 0.0, 50.0], tilt_axis_angle=85.0, ctf=ctf, scale=0.4
    )


def test_projection_ctf():
    ts = _ctf_tilt_series()
    box, spacing = 32, 0.45
    points = np.array([[10.0, -20.0, 30.0], [-5.0, 15.0, -25.0]])
    projected = ts.project_points(points)
    out = ProjectionCTF(ts, ts.ctf)(points, projected, box, spacing)
    assert out.shape == (2, 3, box, box // 2 + 1)
    assert out.dtype == np.float32
    freq = np.hypot(
        sfft.fftfreq(box, spacing)[:, np.newaxis],
        sfft.rfftfreq(box, spacing)[np.newaxis, :],
    )
    depths = ts.point_depths(points)
    for i in range(2):
        for t in range(3):
            # the standard handedness (-1): underfocus is smaller by the depth
            defocus = ts.ctf.defocus_u[t] + depths[i, t] / 1000
            model = CTFModel.from_kv(200.0, 2.7, defocus=defocus)
            assert_allclose(out[i, t], model.simulate(freq), atol=1e-4)
    # the opposite handedness
    inverted = ProjectionCTF(ts, ts.ctf, hand=1)
    assert_allclose(
        inverted.defocus(points), ts.ctf.defocus_u - depths / 1000, atol=1e-12
    )


def test_projection_ctf_filters():
    ts = _ctf_tilt_series()
    box, spacing = 24, 0.4
    points = np.array([[10.0, -20.0, 30.0], [-5.0, 15.0, -25.0]])
    projected = ts.project_points(points)
    correction = ts.ctf_correction()

    def scale_filter(freq_y, freq_x, index):
        return 2.0 + index

    ctf = ProjectionCTF(ts, ts.ctf, filters=[scale_filter, correction])
    out = ctf(points, projected, box, spacing)
    base = ProjectionCTF(ts, ts.ctf)(points, projected, box, spacing)
    freq_y = sfft.fftfreq(box, spacing).astype(np.float32)[:, np.newaxis]
    freq_x = sfft.rfftfreq(box, spacing).astype(np.float32)[np.newaxis, :]
    for i in range(2):
        for t in range(3):
            sign = correction(freq_y, freq_x, t, position=tuple(projected[i, t]))
            assert_allclose(out[i, t], base[i, t] * (2.0 + t) * sign, atol=1e-6)
    # phase flipping at the plane z = 0 makes the first lobe positive
    assert np.all(out[:, :, 1, 0] > 0)


def test_projection_ctf_invalid():
    ts = _ctf_tilt_series()
    with pytest.raises(ValueError):
        ProjectionCTF(ts, ts.ctf, hand=0)
    with pytest.raises(ValueError):
        ProjectionCTF(ts.select_tilts([0, 1]), ts.ctf)
