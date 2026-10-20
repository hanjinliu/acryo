"""Refinement and scoring of the tilt series using particles and their templates."""

from __future__ import annotations

from functools import partial
import math
import threading
from typing import Callable, Literal, NamedTuple, Sequence

import dask
import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft
from scipy import ndimage as ndi
from scipy import optimize

from acryo._types import degree, nm
from acryo.backend._zncc_2d import ncc_2d, zncc_2d, zncc_landscape_2d
from acryo.ctf import TiltSeriesCTF, _CTFPhase
from acryo.projection import (
    LocalShifts2D,
    TiltImageFilter,
    TiltSeries,
    centered_patches_rfft,
)
from acryo.projection._preprocess import _is_position_dependent
from acryo.projection._projector import VolumeProjector, _face_average


class ParticleSet(NamedTuple):
    """Particles of the same kind and their template in the tilt series frame."""

    points: NDArray[np.float64]  # (N, 3) positions relative to the tomogram center (nm)
    rotations: NDArray[np.float64]  # (N, 3, 3) matrices from template to tomogram
    template: NDArray[np.float32]  # (D, H, W) template image


class ProjectionCTF:
    """The CTF of the projections of particles, at the defocus of each particle.

    The defocus of a particle in each tilt is the defocus of ``ctf`` (at the tomogram
    center) changed by the depth of the particle in the microscope relative to the
    tomogram center (``TiltSeries.point_depths``), as
    :class:`acryo.ctf.PatchCTFCorrection`. The projections are then filtered by
    ``filters``, the filters of the preprocessing of the tilt images (such as the CTF
    correction), so that they are comparable with the preprocessed tilt images. The
    position-dependent filters are evaluated at the projected positions of the
    particles.

    Parameters
    ----------
    tilt_series : TiltSeries
        The tilt series, whose geometry gives the depths of the particles.
    ctf : TiltSeriesCTF
        CTF parameters of each tilt.
    hand : {1, -1}, default is -1
        Handedness of the defocus gradient (see :class:`acryo.ctf.PatchCTFCorrection`).
    filters : sequence of TiltImageFilter, optional
        Filters of the preprocessing of the tilt images.
    """

    def __init__(
        self,
        tilt_series: TiltSeries,
        ctf: TiltSeriesCTF,
        hand: int = -1,
        filters: Sequence[TiltImageFilter] = (),
    ):
        if ctf.ntilts != tilt_series.ntilts:
            raise ValueError(
                f"The CTF has the parameters of {ctf.ntilts} tilts, but the tilt "
                f"series has {tilt_series.ntilts} tilts."
            )
        if hand not in (1, -1):
            raise ValueError(f"'hand' must be 1 or -1, got {hand!r}.")
        self._tilt_series = tilt_series
        self._ctf = ctf
        self._hand = int(hand)
        self._filters = tuple(filters)
        self._lock = threading.Lock()
        self._cache: dict[tuple[int, float], _CTFFilters] = {}  # (box, spacing) ->

    def defocus(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Mean defocus (N, T) in μm of the points (N, 3) relative to the center."""
        ctf = self._ctf
        depths = self._tilt_series.point_depths(points)  # (N, T), in nm
        # depth and underfocus are in nm, defocus in μm (negative for underfocus)
        return (ctf.defocus_u + ctf.defocus_v) / 2 - self._hand * depths / 1000

    def __call__(
        self,
        points: NDArray[np.float64],
        projected: NDArray[np.float64],
        box: int,
        spacing: nm,
    ) -> NDArray[np.float32]:
        """Filters (b, T, box, box // 2 + 1) of the rfft of the projections.

        Parameters
        ----------
        points : (b, 3) array
            Positions of the particles relative to the tomogram center in nm.
        projected : (b, T, 2) array
            Projected positions of the particles in nm from the center of the tilt
            images, where the position-dependent filters are evaluated.
        box : int
            Size of the projections.
        spacing : float
            Pixel size of the projections in nm.
        """
        phases, fixed, local = self._for_geometry(box, spacing)
        defocus = self.defocus(points)
        out = np.stack(
            [phase.value(defocus[:, t]) for t, phase in enumerate(phases)], axis=1
        )
        out *= fixed
        for t, funcs in enumerate(local):
            for i in range(out.shape[0]):
                position = (float(projected[i, t, 0]), float(projected[i, t, 1]))
                for func in funcs:
                    out[i, t] *= np.asarray(func(position=position), dtype=np.float32)
        return out

    def _for_geometry(self, box: int, spacing: nm) -> _CTFFilters:
        key = (box, round(spacing, 10))
        with self._lock:
            if (out := self._cache.get(key)) is not None:
                return out
            ctf = self._ctf
            freq_y = sfft.fftfreq(box, spacing).astype(np.float32)[:, np.newaxis]
            freq_x = sfft.rfftfreq(box, spacing).astype(np.float32)[np.newaxis, :]
            phases = [
                _CTFPhase(
                    freq_y,
                    freq_x,
                    astigmatism=(ctf.defocus_u[t] - ctf.defocus_v[t]) / 2,
                    astigmatism_angle=math.radians(ctf.astigmatism_angle[t]),
                    wave_length=ctf.wave_length,
                    spherical_aberration=ctf.spherical_aberration,
                    amplitude_contrast=ctf.amplitude_contrast,
                )
                for t in range(ctf.ntilts)
            ]
            fixed = np.ones((ctf.ntilts, box, box // 2 + 1), dtype=np.float32)
            local: list[list[Callable[..., NDArray[np.floating] | float]]] = []
            for t in range(ctf.ntilts):
                funcs = []
                for filt in self._filters:
                    if not _is_position_dependent(filt):
                        fixed[t] *= np.asarray(filt(freq_y, freq_x, t), np.float32)
                    elif (for_tilt := getattr(filt, "_for_tilt", None)) is not None:
                        funcs.append(for_tilt(freq_y, freq_x, t))
                    else:
                        funcs.append(partial(filt, freq_y, freq_x, t))
                local.append(funcs)
            out = self._cache[key] = _CTFFilters(phases, fixed, local)
        return out


class _CTFFilters(NamedTuple):
    """The CTF and the filters of each tilt for a geometry of the projections."""

    phases: list[_CTFPhase]
    # (T, box, box // 2 + 1) filters that do not depend on the position
    fixed: NDArray[np.float32]
    # position-dependent filters of each tilt as functions of the position
    local: list[list[Callable[..., NDArray[np.floating] | float]]]


def refine_tilt_series(
    tilt_series: TiltSeries,
    images: NDArray[np.float32],
    particle_sets: Sequence[ParticleSet],
    scale: nm,
    max_shift: nm,
    rotation: tuple[degree, degree] = (0.0, 0.0),
    grid_shape: tuple[int, int] = (1, 1),
    smoothness: float = 0.01,
    niter: int = 100,
    batch_size: int = 16,
    ctf: ProjectionCTF | None = None,
) -> TiltSeries:
    """Refine the shifts of the tilt images using particles and their templates.

    For each particle and tilt, the projection of the template is compared with the
    tilt image around the projected position by normalized cross correlation (NCC).
    The shift of each tilt image is the peak of the NCC maps summed over particles.
    If ``rotation`` is given, the NCC maps are computed for each candidate of the
    tilt axis angle, and the angle with the highest peak is chosen for each tilt.
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
    rotation : (float, float), default is (0.0, 0.0)
        (max, step) of the rotation of the tilt images in degrees. The tilt axis
        angle of each tilt is searched from ``-max`` to ``max`` by ``step`` around the
        current one. No rotation is searched if ``max`` or ``step`` is 0.
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
    ctf : ProjectionCTF, optional
        If given, the CTF of each particle is applied to the projections.

    Returns
    -------
    TiltSeries
        A tilt series with refined ``sample_translations`` (``tilt_axis_angle`` if
        ``rotation`` is given, and ``local_shifts_2d`` if ``grid_shape`` is larger
        than ``(1, 1)``, whose base is the local shifts of the input).
    """
    if len(particle_sets) == 0 or all(len(ps.points) == 0 for ps in particle_sets):
        raise ValueError("No particle is given.")
    if len(grid_shape) != 2 or any(g < 1 for g in grid_shape):
        raise ValueError(f"'grid_shape' must be positive, got {grid_shape!r}.")
    if tilt_series.scale is None:
        raise ValueError("The pixel size of the tilt series is not set.")
    angles = _rotation_candidates(rotation)
    input_spacing = tilt_series.scale
    max_shift_px = math.ceil(max_shift / scale)
    cubes = [_centered_cube(ps.template) for ps in particle_sets]
    template_size = max(cube.shape[0] for cube in cubes)
    box, native = _patch_geometry(
        template_size + 2 * (max_shift_px + 1), scale / input_spacing
    )
    spacing = input_spacing * native / box  # true pixel spacing of the patches
    radius = math.ceil(max_shift / spacing)

    image_center = np.array(images.shape[1:]) // 2
    # the templates without the background, which is zero after padding
    projectors = [VolumeProjector(cube - _face_average(cube)) for cube in cubes]

    def _batch(
        ts: TiltSeries,
        rot_detector: NDArray[np.float64],
        ps: ParticleSet,
        projector: VolumeProjector,
        sl: slice,
    ):
        # (b, T, 2) projected positions in nm, including the local shifts
        projected = ts.project_points(ps.points[sl])
        if ctf is None:
            transfer = None
        else:
            transfer = ctf(ps.points[sl], projected, box, spacing)
        maps = _correlation_maps(
            images,
            projected / input_spacing + image_center,
            ps.rotations[sl],
            projector,
            rot_detector,
            box,
            native,
            radius,
            transfer,
        )
        return maps, projected

    def _all_maps(ts: TiltSeries):
        # rotations from the detector to the tomogram, as used in reconstruction
        rot_detector = ts.detector2tomo
        # The arguments are bound by `partial`, so that dask does not traverse the
        # tilt series (a dataclass) and the particle sets.
        tasks = [
            dask.delayed(
                partial(
                    _batch,
                    ts,
                    rot_detector,
                    ps,
                    projector,
                    slice(start, start + batch_size),
                ),
                pure=False,
            )()
            for ps, projector in zip(particle_sets, projectors)
            for start in range(0, len(ps.points), batch_size)
        ]
        results = dask.compute(*tasks)
        maps = np.concatenate([r[0] for r in results])  # (N, T, W, W)
        positions = np.concatenate([r[1] for r in results])  # (N, T, 2), in nm
        return maps, positions

    maps, positions = _all_maps(tilt_series)  # angles[0] is 0
    # For each tilt, keep the maps of the tilt axis angle with the highest peak of
    # the NCC summed over particles.
    best_score = maps.sum(axis=0).max(axis=(-2, -1))  # (T,)
    best_angle = np.zeros(tilt_series.ntilts)
    for angle in angles[1:]:
        rotated = tilt_series.replace(
            tilt_axis_angle=tilt_series.tilt_axis_angle + angle
        )
        maps_a, positions_a = _all_maps(rotated)
        score = maps_a.sum(axis=0).max(axis=(-2, -1))
        better = score > best_score
        maps[:, better] = maps_a[:, better]
        positions[:, better] = positions_a[:, better]
        best_score[better] = score[better]
        best_angle[better] = angle
    if np.any(best_angle != 0):
        tilt_series = tilt_series.replace(
            tilt_axis_angle=tilt_series.tilt_axis_angle + best_angle
        )
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


def score_on_tilt(
    tilt_series: TiltSeries,
    images: NDArray[np.float32],
    points: NDArray[np.float64],
    rotations: NDArray[np.float64],
    templates: Sequence[tuple[NDArray[np.float32], NDArray[np.float32] | None]],
    scale: nm,
    cutoff: float | None = None,
    method: Literal["zncc", "ncc"] = "zncc",
    batch_size: int = 16,
    ctf: ProjectionCTF | None = None,
) -> NDArray[np.float32]:
    """Scores of the particles in each tilt image against the templates.

    For each particle and tilt, the projection of the template is compared with the
    patch of the tilt image at the projected position (no shift is searched) by
    ZNCC or NCC. The patches are of the size of the template (padded to a cube), at
    the pixel size of ``scale``. If a mask is given, it is projected in the same way
    and normalized to the maximum of 1, and multiplied to both the projection and
    the patch.

    Parameters
    ----------
    tilt_series : TiltSeries
        The tilt series, whose ``scale`` is the pixel size of ``images``.
    images : (T, H, W) array
        (Preprocessed) tilt images of the tilt series.
    points : (N, 3) array
        Positions of the particles relative to the tomogram center in nm.
    rotations : (N, 3, 3) array
        Rotation matrices of the particles from the template to the tomogram.
    templates : sequence of (template, mask)
        Templates (bright densities on a dark background) and their masks of the
        same shape, or None for no mask, in the pixel size of ``scale``.
    scale : float
        Pixel size of the templates in nm.
    cutoff : float, optional
        Cutoff frequency of the Butterworth low-pass filter applied to both the
        projections and the patches, in cycles per pixel of ``scale`` (0.5 is the
        Nyquist frequency). No filter by default.
    method : "zncc" or "ncc", default is "zncc"
        Zero-mean normalized cross correlation or normalized cross correlation.
    batch_size : int, default is 16
        Number of particles processed at once.
    ctf : ProjectionCTF, optional
        If given, the CTF of each particle is applied to the projections.

    Returns
    -------
    (n_templates, N, T) array
        ``score[i, j, t]`` is the score of the i-th template for the j-th particle in
        the t-th tilt image.
    """
    if method == "zncc":
        func = zncc_2d
    elif method == "ncc":
        func = ncc_2d
    else:
        raise ValueError(f"'method' must be 'zncc' or 'ncc', got {method!r}.")
    if cutoff is not None and cutoff <= 0:
        raise ValueError(f"'cutoff' must be positive, got {cutoff!r}.")
    if batch_size <= 0:
        raise ValueError(f"'batch_size' must be positive, got {batch_size!r}.")
    if tilt_series.scale is None:
        raise ValueError("The pixel size of the tilt series is not set.")
    input_spacing = tilt_series.scale
    scorers = [
        _TiltScorer(template, mask, scale, input_spacing, cutoff)
        for template, mask in templates
    ]
    npoints = points.shape[0]
    if npoints == 0 or len(scorers) == 0:
        return np.zeros((len(scorers), npoints, tilt_series.ntilts), dtype=np.float32)
    image_center = np.array(images.shape[1:]) // 2
    # rotations from the detector to the tomogram, as used in reconstruction
    rot_detector = tilt_series.detector2tomo

    def _batch(sl: slice) -> NDArray[np.float32]:
        """Scores (n_templates, b, T) of a batch of particles."""
        # (b, T, 2) projected positions in nm, including the local shifts
        projected = tilt_series.project_points(points[sl])
        positions = projected / input_spacing + image_center
        matrices = _template_to_detector(rotations[sl], rot_detector)
        # the templates of the same geometry share the patches and the CTF
        observed: dict[tuple[int, int], NDArray[np.complexfloating]] = {}
        transfers: dict[tuple[int, int], NDArray[np.float32] | None] = {}
        scores = np.empty((len(scorers),) + projected.shape[:2], dtype=np.float32)
        for i, scorer in enumerate(scorers):
            key = (scorer.box, scorer.native)
            if key not in observed:
                observed[key] = _observed_patches(images, positions, *key)
                if ctf is None:
                    transfers[key] = None
                else:
                    transfers[key] = ctf(points[sl], projected, *scorer.geometry)
            scores[i] = scorer.score(observed[key], matrices, func, transfers[key])
        return scores

    tasks = [
        dask.delayed(_batch, pure=False)(slice(start, start + batch_size))
        for start in range(0, npoints, batch_size)
    ]
    return np.concatenate(dask.compute(*tasks), axis=1)


class _TiltScorer:
    """Compare the projections of a template with the patches of the tilt images."""

    def __init__(
        self,
        template: NDArray[np.float32],
        mask: NDArray[np.float32] | None,
        scale: nm,
        input_spacing: nm,
        cutoff: float | None,
    ):
        cube = _centered_cube(template)
        self.size = size = cube.shape[0]
        # twice as large as the template against the wrap-around of the filters
        self.box, self.native = _patch_geometry(2 * size, scale / input_spacing)
        self.spacing = input_spacing * self.native / self.box  # of the patches
        # the template without the background, which is zero after padding
        self.projector = VolumeProjector(cube - _face_average(cube))
        if mask is None:
            self.mask_projector = None
        else:
            if mask.shape != template.shape:
                raise ValueError(
                    f"Shape mismatch between template {template.shape} and mask "
                    f"{mask.shape}."
                )
            self.mask_projector = VolumeProjector(_centered_cube(mask))
        if cutoff is None:
            self.weight = None
        else:
            # the cutoff in cycles per pixel of the patches
            self.weight = _lowpass_weight(self.box, cutoff * self.spacing / scale)

    @property
    def geometry(self) -> tuple[int, nm]:
        """Size and pixel size of the patches."""
        return self.box, self.spacing

    def score(
        self,
        observed: NDArray[np.complexfloating],
        matrices: NDArray[np.float64],
        func: Callable[[NDArray[np.float32], NDArray[np.float32]], NDArray[np.float64]],
        transfer: NDArray[np.float32] | None = None,
    ) -> NDArray[np.float64]:
        """Scores (b, T) of the patches (rfft, origin at index 0).

        ``transfer`` is the filter of the projections, such as the CTF.
        """
        box, size = self.box, self.size
        reference = sfft.rfft2(
            sfft.ifftshift(
                _padded_projections(self.projector, matrices, box), axes=(-2, -1)
            )
        )
        if transfer is not None:
            reference *= transfer
        if self.weight is not None:
            observed = observed * self.weight
            reference = reference * self.weight
        # the region of the projections
        patches = _central_crop(observed, box, size)
        projections = _central_crop(reference, box, size)
        if self.mask_projector is not None:
            mask = self.mask_projector(matrices)  # (b, T, D, D), centered at D // 2
            mask_max = mask.max(axis=(-2, -1), keepdims=True)
            mask = np.clip(mask / np.maximum(mask_max, 1e-12), 0, 1)
            patches = patches * mask
            projections = projections * mask
        return func(patches, projections)


def _central_crop(
    ft: NDArray[np.complexfloating], box: int, size: int
) -> NDArray[np.float64]:
    """Real images (..., size, size) at the center of the rfft (origin at index 0).

    The centers of the cropped images are at ``size // 2``.
    """
    images = sfft.fftshift(sfft.irfft2(ft, s=(box, box)), axes=(-2, -1))
    start = box // 2 - size // 2
    return images[..., start : start + size, start : start + size]


def _lowpass_weight(size: int, cutoff: float, order: int = 2) -> NDArray[np.float64]:
    """Butterworth weights (size, size // 2 + 1) of the rfft of square images.

    ``cutoff`` is in cycles per pixel, as ``acryo.backend.Backend.lowpass_filter``.
    """
    qy, qx = np.meshgrid(sfft.fftfreq(size), sfft.rfftfreq(size), indexing="ij")
    q2 = (qy**2 + qx**2) / cutoff**2
    return 1 / (1 + q2**order)


def _rotation_candidates(rotation: tuple[degree, degree]) -> NDArray[np.float64]:
    """Candidate rotation angles in degrees from (max, step), the smallest first."""
    if len(rotation) != 2:
        raise ValueError(f"'rotation' must be (max, step), got {rotation!r}.")
    max_rot, step = float(rotation[0]), float(rotation[1])
    if max_rot < 0 or step < 0:
        raise ValueError(f"'rotation' must be non-negative, got {rotation!r}.")
    if step == 0:
        return np.zeros(1)
    n = int(max_rot / step + 1e-8)  # tolerate the rounding error such as 0.3 / 0.1
    indices = np.arange(-n, n + 1)
    return indices[np.argsort(np.abs(indices), kind="stable")] * step


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
    transfer: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """ZNCC maps (b, T, 2 * radius + 1, 2 * radius + 1) between particles and images.

    The center of the maps is the projected position (``positions`` in pixel of the
    images) of the particle in the current alignment, and the peak position is the
    shift of the particle in the image. ``transfer`` is the filter of the projections,
    such as the CTF.
    """
    observed = _observed_patches(images, positions, box, native)
    matrices = _template_to_detector(rotations, rot_detector)
    reference = sfft.rfft2(
        sfft.ifftshift(_padded_projections(projector, matrices, box), axes=(-2, -1))
    )
    if transfer is not None:
        reference *= transfer
    return zncc_landscape_2d(observed, reference, radius)


def _observed_patches(
    images: NDArray[np.float32],
    positions: NDArray[np.float64],
    box: int,
    native: int,
) -> NDArray[np.complexfloating]:
    """The rfft (b, T, box, box // 2 + 1) of the patches around the positions.

    The patches of size ``native`` are cropped at the pixel size of the images and
    Fourier-rescaled to ``box``. The positions are at the origin (index 0).
    """
    observed = centered_patches_rfft(images, positions, native)
    return _fourier_resize(observed, box)


def _template_to_detector(
    rotations: NDArray[np.float64], rot_detector: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Matrices (b, T, 3, 3) of the projections of the templates on the detector."""
    return rotations.transpose(0, 2, 1)[:, np.newaxis] @ rot_detector[np.newaxis]


def _padded_projections(
    projector: VolumeProjector, matrices: NDArray[np.float64], box: int
) -> NDArray[np.float32]:
    """Projections (b, T, box, box) of the templates padded by zeros.

    The particles are dark in the tilt images, so the contrast is inverted. The
    centers of the projections are at ``box // 2``.
    """
    projections = -projector(matrices)  # (b, T, D, D), centered at D // 2
    size = projector.size
    pad = box // 2 - size // 2
    padded = np.zeros(projections.shape[:2] + (box, box), dtype=np.float32)
    padded[..., pad : pad + size, pad : pad + size] = projections
    return padded


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
