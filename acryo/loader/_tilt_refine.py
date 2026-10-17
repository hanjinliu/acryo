"""Refinement of the tilt series alignment using particles and their templates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import os
from typing import NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft
from scipy import ndimage as ndi
from scipy import optimize

from acryo._types import nm
from acryo.projection import LocalShifts2D, TiltSeries, centered_patches_rfft
from acryo.projection._projector import VolumeProjector, _face_average


class ParticleSet(NamedTuple):
    """Particles of the same kind and their template in the tilt series frame."""

    points: NDArray[np.float64]  # (N, 3) positions relative to the tomogram center (nm)
    rotations: NDArray[np.float64]  # (N, 3, 3) matrices from template to tomogram
    template: NDArray[np.float32]  # (D, H, W) template image


def refine_tilt_series(
    tilt_series: TiltSeries,
    images: NDArray[np.float32],
    particle_sets: Sequence[ParticleSet],
    scale: nm,
    max_shift: nm,
    grid_shape: tuple[int, int] = (1, 1),
    smoothness: float = 0.01,
    niter: int = 100,
    batch_size: int = 16,
) -> TiltSeries:
    """Refine the shifts of the tilt images using particles and their templates.

    For each particle and tilt, the projection of the template is compared with the
    tilt image around the projected position by normalized cross correlation (NCC).
    The shift of each tilt image is the peak of the NCC maps summed over particles.
    If ``grid_shape`` is larger than ``(1, 1)``, the shifts are modeled as a smooth
    field over each tilt image, interpolated from a grid of control points, and the
    control points are optimized to maximize the NCC with a smoothness prior.

    Parameters
    ----------
    tilt_series : TiltSeries
        The tilt series to be refined, whose ``scale`` is the pixel size of
        ``images``. The projection includes its local shifts.
    images : (T, H, W) array
        (Preprocessed) tilt images of the tilt series.
    particle_sets : sequence of ParticleSet
        Particles and their templates. The templates must be in the pixel size of
        ``scale``, and have bright densities on a dark background.
    scale : float
        Pixel size of the templates in nm.
    max_shift : float
        Maximum shift of the tilt images in nm.
    grid_shape : (int, int), default is (1, 1)
        Number of control points of the local shifts along y and x.
    smoothness : float, default is 0.01
        Weight of the smoothness prior of the local shifts. The penalty is the sum
        of squared differences (nm^2) between the shifts of neighboring control
        points, compared with the sum of NCC over particles.
    niter : int, default is 100
        Maximum number of iterations to optimize the local shifts.
    batch_size : int, default is 16
        Number of particles processed at once.

    Returns
    -------
    TiltSeries
        A tilt series with refined ``sample_translations`` (and ``local_shifts_2d``
        if ``grid_shape`` is larger than ``(1, 1)``, whose base is the local shifts of
        the input).
    """
    if len(particle_sets) == 0 or all(len(ps.points) == 0 for ps in particle_sets):
        raise ValueError("No particle is given.")
    if len(grid_shape) != 2 or any(g < 1 for g in grid_shape):
        raise ValueError(f"'grid_shape' must be positive, got {grid_shape!r}.")
    if tilt_series.scale is None:
        raise ValueError("The pixel size of the tilt series is not set.")
    input_spacing = tilt_series.scale
    max_shift_px = math.ceil(max_shift / scale)
    cubes = [_centered_cube(ps.template) for ps in particle_sets]
    template_size = max(cube.shape[0] for cube in cubes)
    box, native = _patch_geometry(
        template_size + 2 * (max_shift_px + 1), scale / input_spacing
    )
    spacing = input_spacing * native / box  # true pixel spacing of the patches
    radius = math.ceil(max_shift / spacing)

    # rotations from the detector to the tomogram, as used in reconstruction
    rot_detector = tilt_series.detector2tomo
    image_center = np.array(images.shape[1:]) // 2

    def _batch(ps: ParticleSet, projector: VolumeProjector, sl: slice):
        # (b, T, 2) projected positions in nm, including the local shifts
        projected = tilt_series.project_points(ps.points[sl])
        maps = _correlation_maps(
            images,
            projected / input_spacing + image_center,
            ps.rotations[sl],
            projector,
            rot_detector,
            box,
            native,
            radius,
        )
        return maps, projected

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for ps, cube in zip(particle_sets, cubes):
            # the template without the background, which is zero after padding
            projector = VolumeProjector(cube - _face_average(cube))
            for start in range(0, len(ps.points), batch_size):
                sl = slice(start, start + batch_size)
                futures.append(executor.submit(_batch, ps, projector, sl))
        results = [future.result() for future in futures]
    maps = np.concatenate([r[0] for r in results])  # (N, T, W, W)
    positions = np.concatenate([r[1] for r in results])  # (N, T, 2), in nm
    shifts = _subpixel_peak(maps.sum(axis=0))  # (T, 2), in pixel

    if tuple(grid_shape) == (1, 1):
        return tilt_series.replace(
            sample_translations=tilt_series.sample_translations + shifts * spacing
        )

    half_extent = np.array(images.shape[1:]) * input_spacing / 2
    control_points = _fit_local_shifts(
        maps,
        positions / half_extent,
        shifts,
        grid_shape=(int(grid_shape[0]), int(grid_shape[1])),
        smoothness=smoothness * spacing**2,  # per nm^2 to per pixel^2
        niter=niter,
    )
    mean = control_points.mean(axis=(-2, -1))
    return tilt_series.replace(
        sample_translations=tilt_series.sample_translations + mean * spacing,
        local_shifts_2d=LocalShifts2D(
            (control_points - mean[..., np.newaxis, np.newaxis]) * spacing,
            half_extent,
            base=tilt_series.local_shifts_2d,
        ),
    )


def _patch_geometry(min_size: int, ratio: float) -> tuple[int, int]:
    """Choose the patch size and the crop size of the tilt images (both even).

    The tilt images are cropped at their pixel size and Fourier-rescaled to the patch
    size. The patch size is chosen so that the pixel size is changed as little as
    possible by rounding.
    """
    smallest = min_size + min_size % 2

    def _native(box: int) -> int:
        return max(2 * round(box * ratio / 2), 2)

    box = min(
        range(smallest, smallest + 16, 2),
        key=lambda b: abs(_native(b) / (b * ratio) - 1),
    )
    return box, _native(box)


def _centered_cube(template: NDArray[np.float32]) -> NDArray[np.float32]:
    """Pad the template to a cube whose center (size // 2) is the template center.

    The center of the template is ``(shape - 1) / 2``, so even-sized axes are
    shifted by half a pixel.
    """
    template = np.asarray(template, dtype=np.float32)
    shifts = [0.5 if s % 2 == 0 else 0.0 for s in template.shape]
    if any(shifts):
        ft = ndi.fourier_shift(np.fft.fftn(template), shifts)
        template = np.fft.ifftn(ft).real.astype(np.float32)
    size = max(template.shape)
    pads = [
        (size // 2 - s // 2, size - s - (size // 2 - s // 2)) for s in template.shape
    ]
    return np.pad(template, pads)


def _correlation_maps(
    images: NDArray[np.float32],
    positions: NDArray[np.float64],
    rotations: NDArray[np.float64],
    projector: VolumeProjector,
    rot_detector: NDArray[np.float64],
    box: int,
    native: int,
    radius: int,
) -> NDArray[np.float32]:
    """NCC maps (b, T, 2 * radius + 1, 2 * radius + 1) between particles and images.

    The center of the maps is the projected position (``positions`` in pixel of the
    images) of the particle in the current alignment, and the peak position is the
    shift of the particle in the image.
    """
    # observed patches around the projected points, with the origin at index 0
    observed = centered_patches_rfft(images, positions, native)
    observed = _fourier_resize(observed, box)

    # projections of the templates, from the template to the detector
    matrices = rotations.transpose(0, 2, 1)[:, np.newaxis] @ rot_detector[np.newaxis]
    # particles are dark in the tilt images
    projections = -projector(matrices)  # (b, T, D, D), centered at D // 2
    size = projector.size
    pad = box // 2 - size // 2
    padded = np.zeros(projections.shape[:2] + (box, box), dtype=np.float32)
    padded[..., pad : pad + size, pad : pad + size] = projections
    reference = sfft.rfft2(sfft.ifftshift(padded, axes=(-2, -1)))

    # normalized cross correlation, without the mean
    observed[..., 0, 0] = 0
    reference[..., 0, 0] = 0
    norm_obs = _norm(observed, box)
    norm_ref = _norm(reference, box)
    cc = sfft.irfft2(observed * reference.conj(), s=(box, box))
    cc = sfft.fftshift(cc, axes=(-2, -1))
    c = box // 2
    cc = cc[..., c - radius : c + radius + 1, c - radius : c + radius + 1]
    denom = np.maximum(norm_obs * norm_ref, 1e-12)
    return (cc / denom[..., np.newaxis, np.newaxis]).astype(np.float32)


def _norm(ft: NDArray[np.complexfloating], size: int) -> NDArray[np.float64]:
    """L2 norms of the real images of the (..., size, size // 2 + 1) rfft."""
    return np.linalg.norm(sfft.irfft2(ft, s=(size, size)), axis=(-2, -1))


def _fourier_resize(
    ft: NDArray[np.complexfloating], size: int
) -> NDArray[np.complexfloating]:
    """Crop or pad the (..., n, n // 2 + 1) rfft to (..., size, size // 2 + 1).

    The origin of the images is kept at index 0, and the pixel size is changed by the
    factor ``n / size``.
    """
    n = ft.shape[-2]
    if n == size:
        return ft
    m = min(n, size)
    # frequencies (not fftshifted) shared by both sizes
    rows = np.r_[0 : (m + 1) // 2, -(m // 2) : 0]
    out = np.zeros(ft.shape[:-2] + (size, size // 2 + 1), dtype=ft.dtype)
    out[..., rows, : m // 2 + 1] = ft[..., rows, : m // 2 + 1]
    return out


def _subpixel_peak(maps: NDArray[np.floating]) -> NDArray[np.float64]:
    """Peak positions (..., 2) of the maps (..., W, W) relative to the center.

    The peak is refined by parabola fitting along each axis.
    """
    width = maps.shape[-1]
    radius = (width - 1) / 2
    flat = maps.reshape(maps.shape[:-2] + (-1,)).astype(np.float64)
    index = flat.argmax(axis=-1)
    iy, ix = index // width, index % width

    def _value(y, x):
        y = np.clip(y, 0, width - 1)
        x = np.clip(x, 0, width - 1)
        return np.take_along_axis(flat, (y * width + x)[..., np.newaxis], -1)[..., 0]

    def _refine(minus, center, plus, at_edge):
        denom = minus - 2 * center + plus
        delta = 0.5 * (minus - plus) / np.where(denom < 0, denom, -1.0)
        return np.where((denom < 0) & ~at_edge, np.clip(delta, -0.5, 0.5), 0.0)

    center = _value(iy, ix)
    dy = _refine(
        _value(iy - 1, ix),
        center,
        _value(iy + 1, ix),
        (iy == 0) | (iy == width - 1),
    )
    dx = _refine(
        _value(iy, ix - 1),
        center,
        _value(iy, ix + 1),
        (ix == 0) | (ix == width - 1),
    )
    return np.stack([iy + dy, ix + dx], axis=-1) - radius


def _fit_local_shifts(
    maps: NDArray[np.float32],
    positions: NDArray[np.float64],
    init: NDArray[np.float64],
    grid_shape: tuple[int, int],
    smoothness: float,
    niter: int,
) -> NDArray[np.float64]:
    """Optimize the control points of the local shifts of each tilt image.

    For each tilt, the sum of the NCC at the shifts of the particles (bicubic
    interpolation of the maps) minus the smoothness penalty is maximized by L-BFGS-B
    within the maps.

    Parameters
    ----------
    maps : (N, T, W, W) array
        NCC maps of the particles in each tilt image.
    positions : (N, T, 2) array
        (y, x) positions of the particles in each tilt image, normalized to [-1, 1].
    init : (T, 2) array
        Initial shifts of each tilt image in pixel.
    smoothness : float
        Weight of the sum of squared differences (pixel^2) between the shifts of
        neighboring control points, compared with the sum of NCC over particles.
    niter : int
        Maximum number of iterations of L-BFGS-B.

    Returns
    -------
    (T, 2, gy, gx) array
        (y, x) shifts at the control points in pixel.
    """
    _, ntilts, width, _ = maps.shape
    radius = (width - 1) / 2
    gy, gx = grid_shape
    out = np.empty((ntilts, 2, gy, gx))
    for t in range(ntilts):
        nodes, weights = _bilinear_nodes(positions[:, t], grid_shape)
        cc = maps[:, t].astype(np.float64)

        def _loss(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            control = params.reshape(2, gy * gx)
            # (2, N) shifts of the particles
            shift = (control[:, nodes] * weights).sum(axis=-1)
            value, grad_y, grad_x = _bicubic(cc, shift[0] + radius, shift[1] + radius)
            grad = np.stack(
                [
                    np.bincount(
                        nodes.ravel(),
                        (-g[:, np.newaxis] * weights).ravel(),
                        minlength=gy * gx,
                    )
                    for g in (grad_y, grad_x)
                ]
            )
            penalty, grad_penalty = _smoothness_penalty(control.reshape(2, gy, gx))
            loss = smoothness * penalty - value.sum()
            grad += smoothness * grad_penalty.reshape(2, -1)
            return float(loss), grad.ravel()

        x0 = np.repeat(init[t], gy * gx)
        result = optimize.minimize(
            _loss,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-radius, radius)] * x0.size,
            options={"maxiter": niter},
        )
        out[t] = result.x.reshape(2, gy, gx)
    return out


def _bilinear_nodes(
    positions: NDArray[np.float64], grid_shape: tuple[int, int]
) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """Indices (N, 4) of the control points and their bilinear weights.

    The control points are evenly placed from -1 to 1, and the positions outside are
    clamped, as ``LocalShifts2D``.
    """
    axes = []
    for axis, size in enumerate(grid_shape):
        pos = np.clip((positions[:, axis] + 1) / 2 * (size - 1), 0, size - 1)
        i0 = np.minimum(np.floor(pos).astype(np.intp), max(size - 2, 0))
        i1 = np.minimum(i0 + 1, size - 1)
        axes.append((i0, i1, pos - i0))
    (y0, y1, fy), (x0, x1, fx) = axes
    gx = grid_shape[1]
    nodes = np.stack([y0 * gx + x0, y0 * gx + x1, y1 * gx + x0, y1 * gx + x1], axis=1)
    weights = np.stack(
        [(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx], axis=1
    )
    return nodes, weights


def _smoothness_penalty(
    control: NDArray[np.float64],
) -> tuple[float, NDArray[np.float64]]:
    """Sum of squared differences between neighboring control points and its gradient."""
    grad = np.zeros_like(control)
    penalty = 0.0
    for axis in (1, 2):
        diff = np.diff(control, axis=axis)
        penalty += float((diff**2).sum())
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis] = slice(None, -1)
        upper[axis] = slice(1, None)
        grad[tuple(upper)] += 2 * diff
        grad[tuple(lower)] -= 2 * diff
    return penalty, grad


_CUBIC_A = -0.75  # same as the bicubic interpolation of PyTorch


def _cubic_weights(
    frac: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Weights (N, 4) of the cubic convolution at the taps -1, 0, 1, 2 and derivatives."""
    a = _CUBIC_A

    def _inner(x):  # |x| <= 1
        return ((a + 2) * x - (a + 3)) * x * x + 1, (3 * (a + 2) * x - 2 * (a + 3)) * x

    def _outer(x):  # 1 < |x| < 2
        value = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
        return value, (3 * a * x - 10 * a) * x + 8 * a

    w0, d0 = _outer(frac + 1)
    w1, d1 = _inner(frac)
    w2, d2 = _inner(1 - frac)
    w3, d3 = _outer(2 - frac)
    weights = np.stack([w0, w1, w2, w3], axis=-1)
    derivatives = np.stack([d0, d1, -d2, -d3], axis=-1)
    return weights, derivatives


def _bicubic(
    maps: NDArray[np.float64], y: NDArray[np.float64], x: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Bicubic interpolation of each map (N, W, W) at (y, x), and the gradients.

    The taps outside the maps are clamped to the borders.
    """
    n, height, width = maps.shape
    taps = np.arange(-1, 3)
    iy = np.floor(y).astype(np.intp)
    ix = np.floor(x).astype(np.intp)
    wy, dwy = _cubic_weights(y - iy)
    wx, dwx = _cubic_weights(x - ix)
    rows = np.clip(iy[:, np.newaxis] + taps, 0, height - 1)
    cols = np.clip(ix[:, np.newaxis] + taps, 0, width - 1)
    patch = maps[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]]
    value = np.einsum("ni,nij,nj->n", wy, patch, wx)
    grad_y = np.einsum("ni,nij,nj->n", dwy, patch, wx)
    grad_x = np.einsum("ni,nij,nj->n", wy, patch, dwx)
    return value, grad_y, grad_x
