# pyright: reportPrivateImportUsage=false
"""Refinement of the tilt series alignment using particles and their templates."""

from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Callable, NamedTuple, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi

if TYPE_CHECKING:
    import torch
    from torch_tilt_series import TiltSeries


class ParticleSet(NamedTuple):
    """Particles of the same kind and their template in the tilt series frame."""

    points: NDArray[np.float32]  # (N, 3) positions relative to the tomogram center (Å)
    rotations: NDArray[np.float64]  # (N, 3, 3) matrices from template to tomogram
    template: NDArray[np.float32]  # (D, H, W) template image


def refine_tilt_series(
    tilt_series: TiltSeries,
    images: torch.Tensor,
    particle_sets: Sequence[ParticleSet],
    pixel_spacing: float,
    max_shift: float,
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
        The tilt series to be refined. It is not modified.
    images : torch.Tensor
        (Preprocessed) tilt images of the tilt series.
    particle_sets : sequence of ParticleSet
        Particles and their templates. The templates must be in ``pixel_spacing``,
        and have bright densities on a dark background.
    pixel_spacing : float
        Pixel spacing of the templates in Å.
    max_shift : float
        Maximum shift of the tilt images in Å.
    grid_shape : (int, int), default is (1, 1)
        Number of control points of the local shifts along y and x.
    smoothness : float, default is 0.01
        Weight of the smoothness prior of the local shifts. The penalty is the sum
        of squared differences (nm^2) between the shifts of neighboring control
        points, compared with the sum of NCC over particles.
    niter : int, default is 100
        Number of iterations to optimize the local shifts.
    batch_size : int, default is 16
        Number of particles processed at once.

    Returns
    -------
    TiltSeries
        A shallow copy of the tilt series with refined ``sample_translations`` (and
        ``local_shifts_2d`` if ``grid_shape`` is larger than ``(1, 1)``).
    """
    import torch

    if len(particle_sets) == 0 or all(len(ps.points) == 0 for ps in particle_sets):
        raise ValueError("No particle is given.")
    if any(g < 1 for g in grid_shape):
        raise ValueError(f"'grid_shape' must be positive, got {grid_shape!r}.")
    input_spacing = tilt_series.pixel_spacing
    max_shift_px = math.ceil(max_shift / pixel_spacing)
    cubes = [_centered_cube(ps.template) for ps in particle_sets]
    template_size = max(cube.shape[0] for cube in cubes)
    box, native = _patch_geometry(
        template_size + 2 * (max_shift_px + 1), pixel_spacing / input_spacing
    )
    spacing = input_spacing * native / box  # true pixel spacing of the patches
    radius = math.ceil(max_shift / spacing)

    device = images.device
    with torch.no_grad():
        # rotations from the detector to the tomogram, as used in reconstruction
        rot_detector = torch.linalg.pinv(
            tilt_series.projection_matrices[:, :3, :3] @ tilt_series.tomo2sample[:3, :3]
        )
        maps_list: list[torch.Tensor] = []
        positions_list: list[torch.Tensor] = []
        for ps, cube_array in zip(particle_sets, cubes):
            cube = torch.as_tensor(cube_array, device=device)
            for start in range(0, len(ps.points), batch_size):
                sl = slice(start, start + batch_size)
                points = torch.as_tensor(ps.points[sl], device=device).float()
                rotations = torch.as_tensor(ps.rotations[sl], device=device).float()
                maps_list.append(
                    _correlation_maps(
                        tilt_series,
                        images,
                        points,
                        rotations,
                        cube,
                        rot_detector,
                        box,
                        native,
                        radius,
                    )
                )
                positions_list.append(tilt_series.project_points(points))
        maps = torch.cat(maps_list)  # (N, T, W, W)
        positions = torch.cat(positions_list)  # (N, T, 2), in Å
        shifts = _subpixel_peak(maps.sum(dim=0))  # (T, 2), in pixel

    out = copy.copy(tilt_series)
    if tuple(grid_shape) == (1, 1):
        out.sample_translations = tilt_series.sample_translations + shifts * spacing
        return out

    half_extent = torch.tensor(images.shape[-2:], device=device) * input_spacing / 2
    control_points = _fit_local_shifts(
        maps,
        positions / half_extent,
        shifts,
        grid_shape=tuple(grid_shape),
        smoothness=smoothness * (spacing / 10) ** 2,  # per nm^2 to per pixel^2
        niter=niter,
    )
    mean = control_points.mean(dim=(-2, -1))
    out.sample_translations = tilt_series.sample_translations + mean * spacing
    out.local_shifts_2d = LocalShifts2D(
        (control_points - mean[..., None, None]) * spacing,
        half_extent,
        base=tilt_series.local_shifts_2d,
    )
    return out


class LocalShifts2D:
    """Smooth 2D shifts of each tilt image interpolated from a grid of control points.

    This object can be used as ``TiltSeries.local_shifts_2d``.

    Parameters
    ----------
    control_points : (T, 2, gy, gx) tensor
        (y, x) shifts in Å at the control points of each tilt image. The control
        points are evenly placed from edge to edge of the images.
    half_extent : (2,) tensor
        Half size of the tilt images (y, x) in Å.
    base : callable, optional
        Local shifts to be added.
    """

    def __init__(
        self,
        control_points: torch.Tensor,
        half_extent: torch.Tensor,
        base: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.control_points = control_points
        self.half_extent = half_extent
        self.base = base

    def __call__(self, projected: torch.Tensor) -> torch.Tensor:
        """Local shifts of the projected (n_points, n_tilts, 2 or 4) points in Å."""
        import torch.nn.functional as F

        zyxw = projected.shape[-1] == 4
        yx = projected[..., 1:3] if zyxw else projected
        half_extent = self.half_extent.to(yx)
        # (n_tilts, n_points, 1, 2) in (x, y) order for grid_sample
        grid = (yx / half_extent).flip(-1).transpose(0, 1)[:, :, None, :]
        shifts = F.grid_sample(
            self.control_points.to(yx),
            grid,
            align_corners=True,
            padding_mode="border",
        )  # (n_tilts, 2, n_points, 1)
        shifts = shifts[..., 0].permute(2, 0, 1)  # (n_points, n_tilts, 2)
        if zyxw:
            shifts = F.pad(shifts, (1, 1))
        if self.base is not None:
            shifts = shifts + self.base(projected)
        return shifts


def _patch_geometry(min_size: int, ratio: float) -> tuple[int, int]:
    """Choose the patch size and the crop size of the tilt images (both even).

    The tilt images are cropped at the native pixel spacing and Fourier-rescaled to
    the patch size. Odd-sized crops are shifted by one pixel in
    ``torch-subpixel-crop``, so the crop size must be even. The patch size is chosen
    so that the pixel spacing is changed as little as possible by rounding.
    """
    smallest = min_size + min_size % 2

    def _native(box: int) -> int:
        return max(2 * round(box * ratio / 2), 2)

    box = min(
        range(smallest, smallest + 16, 2),
        key=lambda b: abs(_native(b) / (b * ratio) - 1),
    )
    return box, _native(box)


def _centered_cube(
    template: NDArray[np.float32], min_size: int = 10
) -> NDArray[np.float32]:
    """Pad the template to a cube whose center (size // 2) is the template center.

    The center of the template is ``(shape - 1) / 2``, so even-sized axes are
    shifted by half a pixel. ``torch_fourier_slice.project_3d_to_2d`` requires the
    size to be at least 10.
    """
    template = np.asarray(template, dtype=np.float32)
    shifts = [0.5 if s % 2 == 0 else 0.0 for s in template.shape]
    if any(shifts):
        ft = ndi.fourier_shift(np.fft.fftn(template), shifts)
        template = np.fft.ifftn(ft).real.astype(np.float32)
    size = max(max(template.shape), min_size)
    pads = [
        (size // 2 - s // 2, size - s - (size // 2 - s // 2)) for s in template.shape
    ]
    return np.pad(template, pads)


def _correlation_maps(
    tilt_series: TiltSeries,
    images: torch.Tensor,
    points: torch.Tensor,
    rotations: torch.Tensor,
    cube: torch.Tensor,
    rot_detector: torch.Tensor,
    box: int,
    native: int,
    radius: int,
) -> torch.Tensor:
    """NCC maps (b, T, 2 * radius + 1, 2 * radius + 1) between particles and images.

    The center of the maps is the projected position of the particle in the current
    alignment, and the peak position is the shift of the particle in the image.
    """
    import torch
    import torch.nn.functional as F
    from torch_fourier_rescale import fourier_rescale_rfft_2d
    from torch_fourier_slice import project_3d_to_2d
    from torch_reconstruct_tomogram.projection import _extract_particle_tilt_series

    # observed patches around the projected points, with the origin at the center
    observed = _extract_particle_tilt_series(
        tilt_series, images, points, sidelength=native, return_rfft=True
    )  # (b, T, native, native // 2 + 1)
    observed = torch.fft.fftshift(observed, dim=(-2,))
    observed = fourier_rescale_rfft_2d(
        observed, image_shape=(native, native), target_shape=(box, box)
    )
    observed = torch.fft.ifftshift(observed, dim=(-2,))

    # projections of the templates, from the template to the detector
    matrices = rotations.transpose(-1, -2)[:, None] @ rot_detector[None]
    # particles are dark in the tilt images
    projections = -project_3d_to_2d(cube, matrices, zyx_matrices=True)
    size = cube.shape[-1]
    pad = box // 2 - size // 2
    projections = F.pad(projections, [pad, box - size - pad] * 2)
    reference = torch.fft.rfftn(
        torch.fft.ifftshift(projections, dim=(-2, -1)), dim=(-2, -1)
    )

    # normalized cross correlation, without the mean
    observed[..., 0, 0] = 0
    reference[..., 0, 0] = 0
    norm_obs = torch.fft.irfftn(observed, s=(box, box), dim=(-2, -1)).norm(dim=(-2, -1))
    norm_ref = torch.fft.irfftn(reference, s=(box, box), dim=(-2, -1)).norm(
        dim=(-2, -1)
    )
    cc = torch.fft.irfftn(observed * reference.conj(), s=(box, box), dim=(-2, -1))
    cc = torch.fft.fftshift(cc, dim=(-2, -1))
    c = box // 2
    cc = cc[..., c - radius : c + radius + 1, c - radius : c + radius + 1]
    return cc / (norm_obs * norm_ref).clamp_min(1e-12)[..., None, None]


def _subpixel_peak(maps: torch.Tensor) -> torch.Tensor:
    """Peak positions (..., 2) of the maps (..., W, W) relative to the center.

    The peak is refined by parabola fitting along each axis.
    """
    import torch

    width = maps.shape[-1]
    radius = (width - 1) / 2
    flat = maps.flatten(-2)
    index = flat.argmax(dim=-1)
    iy, ix = index // width, index % width

    def _value(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        y = y.clamp(0, width - 1)
        x = x.clamp(0, width - 1)
        return flat.gather(-1, (y * width + x)[..., None])[..., 0]

    def _refine(minus, center, plus, at_edge) -> torch.Tensor:
        denom = minus - 2 * center + plus
        delta = 0.5 * (minus - plus) / torch.where(denom < 0, denom, -1.0)
        return torch.where((denom < 0) & ~at_edge, delta.clamp(-0.5, 0.5), 0.0)

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
    return torch.stack([iy + dy, ix + dx], dim=-1) - radius


def _fit_local_shifts(
    maps: torch.Tensor,
    positions: torch.Tensor,
    init: torch.Tensor,
    grid_shape: tuple[int, int],
    smoothness: float,
    niter: int,
) -> torch.Tensor:
    """Optimize the control points of the local shifts of each tilt image.

    Parameters
    ----------
    maps : (N, T, W, W) tensor
        NCC maps of the particles in each tilt image.
    positions : (N, T, 2) tensor
        (y, x) positions of the particles in each tilt image, normalized to [-1, 1].
    init : (T, 2) tensor
        Initial shifts of each tilt image in pixel.
    smoothness : float
        Weight of the sum of squared differences (pixel^2) between the shifts of
        neighboring control points, compared with the sum of NCC over particles.

    Returns
    -------
    (T, 2, gy, gx) tensor
        (y, x) shifts at the control points in pixel.
    """
    import torch
    import torch.nn.functional as F

    nparticles, ntilts, width, _ = maps.shape
    radius = (width - 1) / 2
    cc = maps.transpose(0, 1).reshape(ntilts * nparticles, 1, width, width)
    # (T, N, 1, 2) in (x, y) order for grid_sample
    grid = positions.transpose(0, 1).flip(-1)[:, :, None, :]
    control_points = init[:, :, None, None].repeat(1, 1, *grid_shape)
    control_points.requires_grad_(True)
    optimizer = torch.optim.Adam([control_points], lr=0.1)
    for _ in range(niter):
        optimizer.zero_grad()
        shifts = F.grid_sample(
            control_points, grid, align_corners=True, padding_mode="border"
        )  # (T, 2, N, 1)
        shifts = shifts[..., 0].transpose(1, 2)  # (T, N, 2)
        loc = (shifts / radius).flip(-1).reshape(ntilts * nparticles, 1, 1, 2)
        score = F.grid_sample(
            cc, loc, mode="bicubic", align_corners=True, padding_mode="border"
        )
        score = score.reshape(ntilts, nparticles).sum(dim=1)
        penalty = (control_points.diff(dim=-2) ** 2).sum(dim=(1, 2, 3)) + (
            control_points.diff(dim=-1) ** 2
        ).sum(dim=(1, 2, 3))
        loss = (smoothness * penalty - score).sum()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            control_points.clamp_(-radius, radius)
    return control_points.detach()
