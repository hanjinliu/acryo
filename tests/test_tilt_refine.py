import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import fft as sfft

from acryo.loader._tilt_refine import (
    _bicubic,
    _bilinear_nodes,
    _fourier_resize,
    _smoothness_penalty,
    _subpixel_peak,
)
from acryo.projection import LocalShifts2D


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


def test_bicubic_matches_torch():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    rng = np.random.default_rng(1)
    maps = rng.normal(size=(20, 9, 9))
    y, x = rng.uniform(0, 8, size=(2, 20))
    grid = np.stack([x, y], axis=-1) / 8 * 2 - 1  # normalized (x, y)
    ref = F.grid_sample(
        torch.as_tensor(maps)[:, None],
        torch.as_tensor(grid)[:, None, None],
        mode="bicubic",
        align_corners=True,
        padding_mode="border",
    )
    assert_allclose(_bicubic(maps, y, x)[0], ref.numpy().ravel(), atol=1e-10)


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
