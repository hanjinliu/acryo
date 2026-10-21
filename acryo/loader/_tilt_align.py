"""Alignment of the tilt series of large errors using particles and their templates."""

from __future__ import annotations

import math
from typing import Callable, NamedTuple, Sequence
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy import optimize

from acryo._types import degree, nm
from acryo.loader._tilt_refine import (
    ParticleSet,
    ProjectionCTF,
    _MapGeometry,
    _bicubic,
    _particle_maps,
    _subpixel_peak,
)
from acryo.projection import TiltSeries

# The correlation maps are padded by zeros, so that the particles are not pulled out of
# the maps.
_PAD = 2
# Maximum rotation of the tilt images in each iteration.
_MAX_ANGLE = math.radians(10.0)
# Maximum number of iterations of L-BFGS-B, after adding each tilt and at the end.
_MAXITER_STEP = 30
_MAXITER = 200
# The iterations stop if the particles move less than this (pixel of the maps).
_TOLERANCE = 0.1
# The shift of a tilt is not searched along the direction in which the curvature of
# the summed correlation is less than this fraction of the largest curvature.
_MIN_CURVATURE = 0.1


class TiltAlignment(NamedTuple):
    """The result of :func:`align_tilt_series`."""

    tilt_series: TiltSeries
    offsets: NDArray[np.float64]  # (G, 3) offsets of the particle groups in nm
    scores: NDArray[np.float32]  # (N, T) correlation of the particles


def align_tilt_series(
    tilt_series: TiltSeries,
    images: NDArray[np.float32],
    particle_sets: Sequence[ParticleSet],
    groups: NDArray[np.intp],
    scale: nm,
    max_shift: nm,
    cutoff: float | None = None,
    position_sigma: nm | None = 10.0,
    fit_angle: bool = True,
    max_tilt_angle_change: degree = 0.0,
    progressive: bool = True,
    niter: int = 3,
    batch_size: int = 16,
    ctf: ProjectionCTF | None = None,
) -> TiltAlignment:
    """Align the tilt series of large errors using particles and their templates.

    For each particle and tilt, the projection of the template is compared with the
    tilt image around the projected position by ZNCC within ``max_shift``. The shifts
    of the tilt images, the rotation of all the tilt images (tilt axis angle), the tilt
    angles and the 3D offsets of the groups of particles are optimized together to
    maximize the sum of the ZNCC of all the particles in all the tilt images, as the
    alignment of a tilt series using fiducial markers. The mean offset of the particles
    and the mean change of the tilt angles are fixed to zero. The ZNCC is computed for
    the alignment at the beginning of each iteration, and the projections of the
    templates are not updated by the corrections during the optimization.

    In the first iteration, the tilts are added in the order of their absolute tilt
    angles if ``progressive``. The ZNCC of each added tilt is computed around the
    positions predicted by the current rotation and offsets and the shift of the
    nearest tilt already added, so that the errors accumulating over the tilts can be
    larger than ``max_shift``. The shift is searched for the peak of the ZNCC summed
    over the particles, and then the parameters of the added tilts are optimized.
    Otherwise, the shift of each tilt is searched independently. The ZNCC is computed
    again at the updated alignment in each of the next iterations.

    Parameters
    ----------
    tilt_series : TiltSeries
        The tilt series to be aligned, whose ``scale`` is the pixel size of
        ``images``.
    images : (T, H, W) array
        (Preprocessed) tilt images of the tilt series.
    particle_sets : sequence of ParticleSet
        Particles and their templates. The templates must be in the pixel size of
        ``scale``, and have bright densities on a dark background.
    groups : (N,) array of int
        Group index of each particle of all the particle sets (concatenated). The
        particles of the same group share the 3D offset.
    scale : float
        Pixel size of the templates in nm.
    max_shift : float
        Maximum shift of the particles from the predicted positions in nm.
    cutoff : float, optional
        Cutoff frequency of the low-pass filter applied to both the projections and
        the patches, in cycles per pixel of ``scale``.
    position_sigma : float, optional
        Standard deviation of the prior of the offsets of the particles in nm. The
        penalty of each particle is ``|offset|^2 / (2 * position_sigma^2)``, compared
        with its ZNCC summed over the tilts. If None, the particles are not moved.
    fit_angle : bool, default is True
        Whether to optimize the rotation of all the tilt images.
    max_tilt_angle_change : float, default is 0.0
        Maximum change of each tilt angle in degrees from the input. The tilt angles
        are not optimized if 0.
    progressive : bool, default is True
        Whether to add the tilts one by one in the first iteration.
    niter : int, default is 3
        Maximum number of iterations of computing the maps and optimizing the
        parameters.
    batch_size : int, default is 16
        Number of particles processed at once.
    ctf : ProjectionCTF, optional
        If given, the CTF of each particle is applied to the projections.
    """
    if len(particle_sets) == 0 or all(len(ps.points) == 0 for ps in particle_sets):
        raise ValueError("No particle is given.")
    if tilt_series.scale is None:
        raise ValueError("The pixel size of the tilt series is not set.")
    if max_shift <= 0:
        raise ValueError(f"'max_shift' must be positive, got {max_shift!r}.")
    if cutoff is not None and cutoff <= 0:
        raise ValueError(f"'cutoff' must be positive, got {cutoff!r}.")
    if position_sigma is not None and position_sigma <= 0:
        raise ValueError(f"'position_sigma' must be positive, got {position_sigma!r}.")
    if max_tilt_angle_change < 0:
        raise ValueError(
            "'max_tilt_angle_change' must be non-negative, got "
            f"{max_tilt_angle_change!r}."
        )
    if niter < 1:
        raise ValueError(f"'niter' must be positive, got {niter!r}.")
    if batch_size <= 0:
        raise ValueError(f"'batch_size' must be positive, got {batch_size!r}.")
    npoints = sum(len(ps.points) for ps in particle_sets)
    groups = np.asarray(groups, dtype=np.intp)
    if groups.shape != (npoints,) or groups.min() < 0:
        raise ValueError(f"'groups' must be ({npoints},) non-negative integers.")
    geometry = _MapGeometry.create(particle_sets, scale, tilt_series.scale, max_shift)
    spacing = geometry.spacing
    weight = geometry.lowpass(cutoff, scale)
    projectors = [geometry.projector(ps.template) for ps in particle_sets]
    if position_sigma is None:
        prior = None
    else:
        prior = (spacing / position_sigma) ** 2  # per pixel^2 of the offsets
    starts = np.cumsum([0] + [len(ps.points) for ps in particle_sets])
    points = np.concatenate([ps.points for ps in particle_sets])  # (N, 3) in nm
    offsets = np.zeros((int(groups.max()) + 1, 3))  # in nm
    tilt_changes = np.zeros(tilt_series.ntilts)  # in radian

    for i in range(niter):
        moved = points + offsets[groups]
        moved_sets = [
            ps._replace(points=moved[s0:s1])
            for ps, s0, s1 in zip(particle_sets, starts[:-1], starts[1:])
        ]

        def _maps(
            tilts: Sequence[int] | None = None,
            displacements: NDArray[np.float64] | None = None,
            ts: TiltSeries = tilt_series,
            moved_sets: list[ParticleSet] = moved_sets,
        ) -> NDArray[np.float32]:
            return _particle_maps(
                ts,
                images,
                moved_sets,
                projectors,
                geometry,
                batch_size,
                ctf,
                weight,
                tilts=tilts,
                displacements=displacements,
            )[0]

        scope, linear, detector = _microscope_frame(tilt_series, moved)
        problem = _AlignmentProblem(
            scope / spacing,
            linear,
            detector,
            groups,
            radius=geometry.radius,
            prior=prior,
            offsets=offsets / spacing,
            fit_angle=fit_angle,
            max_tilt=math.radians(max_tilt_angle_change),
            tilt_changes=tilt_changes,
        )
        if i == 0 and progressive:
            order = np.argsort(np.abs(tilt_series.tilt_angles), kind="stable")
            corrections = problem.search_progressive(
                order,
                _nearest_preceding(tilt_series.tilt_angles, order),
                lambda t, disp: _maps([t], disp[:, np.newaxis] * spacing)[:, 0],
            )
        else:
            problem.set_maps(_maps())
            if i == 0:
                corrections = problem.search_independent()
            else:
                corrections = problem.zeros()
                problem.find_flat_directions(corrections)
                corrections = problem.optimize(corrections, problem.all_tilts)
        scores = problem.scores(corrections)
        tilt_series = tilt_series.replace(
            sample_translations=tilt_series.sample_translations
            + corrections.shifts * spacing,
            tilt_axis_angle=tilt_series.tilt_axis_angle
            + math.degrees(corrections.angle),
            tilt_angles=tilt_series.tilt_angles + np.degrees(corrections.tilt_angles),
        )
        offsets = offsets + corrections.offsets * spacing
        tilt_changes = tilt_changes + corrections.tilt_angles
        displacement = np.linalg.norm(problem.displacements(corrections), axis=-1)
        if displacement.max() < _TOLERANCE:
            break
    if (nlimit := problem.tilts_at_limit.size) > 0:
        warnings.warn(
            f"The changes of {nlimit} tilt angles reached the limit of "
            f"±{max_tilt_angle_change:g}°. The tilt angles may not be determined by "
            "the particles, such as when the particles are few or at similar depths, "
            "or 'max_tilt_angle_change' is too small.",
            UserWarning,
            stacklevel=3,
        )
    return TiltAlignment(tilt_series, offsets, scores)


def _microscope_frame(
    tilt_series: TiltSeries, points: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The points in the microscope frame of each tilt, and the matrices.

    The microscope frame is the sample space rotated by the tilt angles (``x_tilts``
    and ``tilt_angles``), whose (z, y, x) are the depth and the position on the
    detector before the rotation of the tilt axis and the shift of the tilt image.

    Returns
    -------
    (N, T, 3) array
        The points (N, 3) relative to the tomogram center in the microscope frame of
        each tilt, including the local shifts of the sample.
    (T, 3, 3) array
        The linear map from the tomogram to the microscope frame of each tilt.
    (T, 2, 2) array
        The rotation of (y, x) of the microscope frame by the tilt axis angle.
    """
    tomo2sample = tilt_series.tomo2sample
    sample = points @ tomo2sample[:3, :3].T + tomo2sample[:3, 3]
    if tilt_series.local_shifts is not None:
        sample = sample + tilt_series.local_shifts(sample)
    rotation = tilt_series.sample2scope[:, :3, :3]
    scope = np.einsum("tij,nj->nti", rotation, sample)
    linear = rotation @ tomo2sample[:3, :3]
    return scope, linear, tilt_series.scope2detector[:, 1:3, 1:3]


def _nearest_preceding(
    angles: NDArray[np.float64], order: NDArray[np.intp]
) -> NDArray[np.intp]:
    """The tilt nearest in angle among the preceding tilts in the order.

    The first tilt has no preceding tilt and refers to itself.
    """
    out = np.empty(order.size, dtype=np.intp)
    out[0] = order[0]
    for k in range(1, order.size):
        preceding = order[:k]
        out[k] = preceding[np.argmin(np.abs(angles[preceding] - angles[order[k]]))]
    return out


class _Corrections(NamedTuple):
    """Corrections of the alignment relative to the current alignment."""

    shifts: NDArray[np.float64]  # (T, 2) shifts of the tilt images in pixel
    angle: float  # rotation of all the tilt images in radian
    tilt_angles: NDArray[np.float64]  # (T,) changes of the tilt angles in radian
    offsets: NDArray[np.float64]  # (G, 3) offsets of the particle groups in pixel


class _AlignmentProblem:
    """Maximize the correlation of the particles over the corrections of alignment.

    The position of the n-th particle in the microscope frame of the t-th tilt is
    ``v = v0[n, t] + L[t] @ offsets[g[n]]``, which is rotated by the change of the tilt
    angle ``e[t]`` around the y axis as ``w = (v_y, sin(e) v_z + cos(e) v_x)``. The
    displacement of the particle in the tilt image from the current alignment is
    ``R @ D[t] @ w - D[t] @ v0_yx + shifts[t]``, where ``D`` is the rotation by the tilt
    axis angle and ``R`` is the rotation of all the tilt images. The correlation is the
    value of the map at the displacement from the center of the map. The maps (and the
    projections of the templates) are not changed by the corrections.

    The weighted mean of the offsets is fixed to zero, as the shifts of the tilt images
    can compensate a common offset. The mean change of the tilt angles (of the tilts in
    the optimization) is also fixed to zero, as it is a rotation of the tomogram around
    the tilt axis. The shift of each tilt image is kept at the center of its maps by a
    prior along the direction in which the ZNCC summed over particles is flat (see
    :meth:`_consensus`), such as along parallel filaments.

    Parameters
    ----------
    scope : (N, T, 3) array
        Positions of the particles in the microscope frame of each tilt (see
        :func:`_microscope_frame`) in pixel of the maps.
    linear : (T, 3, 3) array
        Linear map from the tomogram to the microscope frame of each tilt.
    detector : (T, 2, 2) array
        Rotation of (y, x) by the tilt axis angle of each tilt.
    groups : (N,) array of int
        Group index of each particle.
    radius : int
        Radius of the maps in pixel.
    prior : float, optional
        Weight of the squared offset (pixel^2) of each particle. If None, the offsets
        are fixed to zero.
    offsets : (G, 3) array
        Offsets of the groups already applied to the particles, in pixel. The prior is
        of the total offsets.
    fit_angle : bool
        Whether to optimize the rotation of the tilt images.
    max_tilt : float, default is 0.0
        Maximum change of the tilt angles in radian, including ``tilt_changes``. Not
        optimized if 0.
    tilt_changes : (T,) array, optional
        Changes of the tilt angles already applied, in radian.
    """

    def __init__(
        self,
        scope: NDArray[np.float64],
        linear: NDArray[np.float64],
        detector: NDArray[np.float64],
        groups: NDArray[np.intp],
        radius: int,
        prior: float | None,
        offsets: NDArray[np.float64],
        fit_angle: bool,
        max_tilt: float = 0.0,
        tilt_changes: NDArray[np.float64] | None = None,
    ):
        npoints, ntilts = scope.shape[:2]
        # the projected positions before the shifts of the tilt images
        projected = np.einsum("tij,ntj->nti", detector, scope[..., 1:])
        width = 2 * radius + 1
        self._radius = radius
        # ZNCC maps of the particles, and the ones padded by zeros without negative
        # values, so that negative correlation does not pull the particles
        self._raw = np.zeros((npoints, ntilts, width, width), dtype=np.float32)
        self._maps = np.zeros(
            (npoints, ntilts, width + 2 * _PAD, width + 2 * _PAD), dtype=np.float32
        )
        # (N, T, 2) displacements at the centers of the maps, and (T, 2) the shifts of
        # the tilt images at the centers of the maps in pixel
        self._centers = np.zeros((npoints, ntilts, 2))
        self._shift_centers = np.zeros((ntilts, 2))
        # (T, 2, 2) weights of the prior of the shifts along the flat directions
        self._flat = np.zeros((ntilts, 2, 2))
        self._scope = scope
        self._projected = projected
        self._linear = linear
        self._detector = detector
        self._groups = groups
        counts = np.bincount(groups, minlength=offsets.shape[0])
        self._counts = counts.astype(np.float64)  # weights of the groups
        self._prior = prior
        self._offsets = offsets
        self._fit_angle = fit_angle
        self._max_tilt = max_tilt
        if tilt_changes is None:
            tilt_changes = np.zeros(ntilts)
        self._tilt_changes = tilt_changes
        # tilts whose angles reached the limit in the last optimization
        self.tilts_at_limit = np.zeros(0, dtype=np.intp)
        # The rotations are optimized as the displacements at the typical distances
        # from the axes, so that they are of the same order as the shifts.
        rms = float(np.sqrt((projected**2).sum(axis=-1).mean()))
        self._angle_unit = 1.0 / max(rms, 1.0)
        depth = float(np.sqrt((scope[..., 0] ** 2).mean()))
        self._tilt_unit = 1.0 / max(depth, 1.0)

    @property
    def ntilts(self) -> int:
        return self._projected.shape[1]

    @property
    def ngroups(self) -> int:
        return self._offsets.shape[0]

    @property
    def all_tilts(self) -> NDArray[np.intp]:
        return np.arange(self.ntilts)

    def zeros(self) -> _Corrections:
        """No correction."""
        return _Corrections(
            np.zeros((self.ntilts, 2)),
            0.0,
            np.zeros(self.ntilts),
            np.zeros((self.ngroups, 3)),
        )

    def set_maps(
        self,
        maps: NDArray[np.float32],
        tilt: int | None = None,
        centers: NDArray[np.float64] | None = None,
        shift: NDArray[np.float64] | None = None,
    ) -> None:
        """Set the maps (N, T, W, W) of all the tilts or (N, W, W) of a tilt.

        ``centers`` (N, 2) are the displacements of the particles at the centers of
        the maps of the tilt, and ``shift`` is the shift of the tilt image there.
        """
        index = slice(None) if tilt is None else tilt
        inner = (slice(_PAD, -_PAD),) * 2
        self._raw[:, index] = maps
        self._maps[(slice(None), index) + inner] = np.maximum(maps, 0)
        if centers is not None:
            self._centers[:, index] = centers
        if shift is not None:
            self._shift_centers[index] = shift

    def displacements(
        self, corrections: _Corrections, tilts: Sequence[int] | None = None
    ) -> NDArray[np.float64]:
        """Displacements (N, Ta, 2) of the particles in the tilts in pixel."""
        return self._displace(corrections, tilts)[2]

    def scores(self, corrections: _Corrections) -> NDArray[np.float32]:
        """ZNCC (N, T) of the particles at the corrected alignment."""
        npoints, ntilts = self._projected.shape[:2]
        disp = self.displacements(corrections) - self._centers + self._radius
        value, _, _ = _bicubic(
            self._raw.reshape(npoints * ntilts, *self._raw.shape[2:]),
            disp[..., 0].ravel(),
            disp[..., 1].ravel(),
        )
        return value.reshape(npoints, ntilts).astype(np.float32)

    def find_flat_directions(self, corrections: _Corrections) -> None:
        """Find the flat directions of the summed ZNCC at the current shifts."""
        for t in range(self.ntilts):
            total, center = self._summed_map(corrections, t)
            self._flat[t] = _split_directions(total, corrections.shifts[t] - center)[1]

    def search_independent(self) -> _Corrections:
        """Search the shifts of all the tilts independently, and optimize them."""
        corrections = self.zeros()
        for t in range(self.ntilts):
            corrections.shifts[t] = self._consensus(corrections, t)
        return self.optimize(corrections, self.all_tilts)

    def search_progressive(
        self,
        order: NDArray[np.intp],
        preceding: NDArray[np.intp],
        compute_maps: Callable[[int, NDArray[np.float64]], NDArray[np.float32]],
    ) -> _Corrections:
        """Add the tilts in the order, searching and optimizing the shifts.

        The maps of each added tilt are computed by ``compute_maps(t, displacements)``
        around the displacements (N, 2) in pixel predicted from the current corrections
        and the shift of the ``preceding`` tilt.
        """
        corrections = self.zeros()
        active = np.zeros(self.ntilts, dtype=bool)
        for t, prev in zip(order, preceding):
            shift = corrections.shifts[prev].copy()
            corrections.shifts[t] = shift
            centers = self.displacements(corrections, [t])[:, 0]
            self.set_maps(compute_maps(t, centers), t, centers, shift)
            corrections.shifts[t] = self._consensus(corrections, t)
            active[t] = True
            corrections = self.optimize(
                corrections, np.flatnonzero(active), maxiter=_MAXITER_STEP
            )
        return self.optimize(corrections, self.all_tilts)

    def optimize(
        self,
        init: _Corrections,
        tilts: NDArray[np.intp],
        maxiter: int = _MAXITER,
    ) -> _Corrections:
        """Optimize the corrections using the maps of the given tilts.

        The shifts of the other tilts are kept, and their tilt angles are not changed.
        """
        radius = self._radius
        active = np.zeros(self.ntilts, dtype=bool)
        active[tilts] = True
        bounds: list[tuple[float | None, float | None]] = []
        for t in range(self.ntilts):
            for value, center in zip(init.shifts[t], self._shift_centers[t]):
                if active[t]:
                    bounds.append((center - radius, center + radius))
                else:
                    bounds.append((value, value))
        if self._fit_angle:
            max_angle = min(float(radius), _MAX_ANGLE / self._angle_unit)
            bounds.append((-max_angle, max_angle))
        else:
            bounds.append((0.0, 0.0))
        # changes of the tilt angles within the limit including the applied ones
        tilt_bounds = np.zeros((self.ntilts, 2))
        if self._max_tilt > 0:
            tilt_bounds[active, 0] = -self._max_tilt - self._tilt_changes[active]
            tilt_bounds[active, 1] = self._max_tilt - self._tilt_changes[active]
        tilt_bounds /= self._tilt_unit
        bounds += [(float(lo), float(hi)) for lo, hi in tilt_bounds]
        if self._prior is None:
            bounds += [(0.0, 0.0)] * (3 * self.ngroups)
        else:
            bounds += [(-radius, radius)] * (3 * self.ngroups)
        result = optimize.minimize(
            self._loss,
            self._pack(init),
            args=(tilts,),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": maxiter},
        )
        if self._max_tilt > 0:
            ntilts = self.ntilts
            tilt_params = result.x[2 * ntilts + 1 : 3 * ntilts + 1, np.newaxis]
            at_limit = np.isclose(tilt_params, tilt_bounds, rtol=0, atol=1e-6).any(1)
            self.tilts_at_limit = np.flatnonzero(active & at_limit)
        return self._unpack(result.x, tilts)

    def _pack(self, corrections: _Corrections) -> NDArray[np.float64]:
        return np.concatenate(
            [
                corrections.shifts.ravel(),
                [corrections.angle / self._angle_unit],
                corrections.tilt_angles / self._tilt_unit,
                corrections.offsets.ravel(),
            ]
        )

    def _unpack(
        self, params: NDArray[np.float64], tilts: NDArray[np.intp] | None = None
    ) -> _Corrections:
        """The corrections of the parameters, optimized with the tilts."""
        ntilts = self.ntilts
        shifts = params[: 2 * ntilts].reshape(ntilts, 2).copy()
        angle = float(params[2 * ntilts]) * self._angle_unit
        tilt_angles = params[2 * ntilts + 1 : 3 * ntilts + 1] * self._tilt_unit
        free = params[3 * ntilts + 1 :].reshape(-1, 3)
        return _Corrections(
            shifts,
            angle,
            self._fix_tilt_gauge(tilt_angles, tilts),
            self._fix_gauge(free),
        )

    def _fix_tilt_gauge(
        self, values: NDArray[np.float64], tilts: NDArray[np.intp] | None
    ) -> NDArray[np.float64]:
        """Subtract the mean over the tilts, and zero the others.

        This is symmetric, so the same is applied to the gradient.
        """
        if tilts is None:
            tilts = self.all_tilts
        out = np.zeros(self.ntilts)
        out[tilts] = values[tilts] - values[tilts].mean()
        return out

    def _fix_gauge(self, offsets: NDArray[np.float64]) -> NDArray[np.float64]:
        """Subtract the weighted mean of the offsets."""
        counts = self._counts
        mean = counts @ offsets / counts.sum()
        return offsets - mean

    def _fix_gauge_grad(self, grad: NDArray[np.float64]) -> NDArray[np.float64]:
        """Gradient with respect to the offsets before fixing the gauge.

        It is the transpose of :meth:`_fix_gauge` applied to the gradient with respect
        to the offsets after fixing the gauge.
        """
        counts = self._counts
        return grad - counts[:, np.newaxis] / counts.sum() * grad.sum(axis=0)

    def _displace(
        self, corrections: _Corrections, tilts: Sequence[int] | None = None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """The displacements (N, Ta, 2) in pixel and the intermediate positions.

        Returns the positions (N, Ta, 3) in the microscope frame, the positions
        (N, Ta, 2) on the detector before the rotation of all the tilt images, and the
        displacements.
        """
        if tilts is None:
            tilts = self.all_tilts
        scope = self._scope[:, tilts] + np.einsum(
            "tij,nj->nti", self._linear[tilts], corrections.offsets[self._groups]
        )
        tilt = corrections.tilt_angles[tilts]
        cos, sin = np.cos(tilt), np.sin(tilt)
        tilted = np.stack(
            [scope[..., 1], sin * scope[..., 0] + cos * scope[..., 2]], axis=-1
        )
        detected = np.einsum("tij,ntj->nti", self._detector[tilts], tilted)
        rot, _ = _rotation_2d(corrections.angle)
        disp = detected @ rot.T - self._projected[:, tilts] + corrections.shifts[tilts]
        return scope, detected, disp

    def _loss(
        self, params: NDArray[np.float64], tilts: NDArray[np.intp]
    ) -> tuple[float, NDArray[np.float64]]:
        """Negative sum of the correlation with the prior, and its gradient."""
        corrections = self._unpack(params, tilts)
        scope, detected, disp = self._displace(corrections, tilts)
        npoints = scope.shape[0]
        # positions in the padded maps
        pos = disp - self._centers[:, tilts] + self._radius + _PAD
        index = (np.arange(npoints)[:, np.newaxis] * self.ntilts + tilts).ravel()
        value, grad_y, grad_x = _bicubic(
            self._maps.reshape(npoints * self.ntilts, *self._maps.shape[2:]),
            pos[..., 0].ravel(),
            pos[..., 1].ravel(),
            index,
        )
        # gradient of the loss with respect to the displacements
        grad = -np.stack([grad_y, grad_x], axis=-1).reshape(disp.shape)
        grad_shifts = np.zeros((self.ntilts, 2))
        grad_shifts[tilts] = grad.sum(axis=0)
        rot, drot = _rotation_2d(corrections.angle)
        grad_angle = float((grad * (detected @ drot.T)).sum()) * self._angle_unit
        # gradients with respect to the positions before the rotations
        grad_tilted = np.einsum("nti,tij->ntj", grad @ rot, self._detector[tilts])
        tilt = corrections.tilt_angles[tilts]
        cos, sin = np.cos(tilt), np.sin(tilt)
        grad_tilt = np.zeros(self.ntilts)
        grad_tilt[tilts] = (
            grad_tilted[..., 1] * (cos * scope[..., 0] - sin * scope[..., 2])
        ).sum(axis=0)
        grad_scope = np.stack(
            [grad_tilted[..., 1] * sin, grad_tilted[..., 0], grad_tilted[..., 1] * cos],
            axis=-1,
        )
        grad_points = np.einsum("ntk,tkj->nj", grad_scope, self._linear[tilts])
        grad_offsets = np.stack(
            [
                np.bincount(self._groups, grad_points[:, k], minlength=self.ngroups)
                for k in range(3)
            ],
            axis=1,
        )
        loss = -float(value.sum())
        if self._prior is not None:
            total = self._offsets + corrections.offsets
            loss += 0.5 * self._prior * float(self._counts @ (total**2).sum(axis=1))
            grad_offsets += self._prior * self._counts[:, np.newaxis] * total
        # the prior of the shifts along the flat directions
        diff = corrections.shifts[tilts] - self._shift_centers[tilts]
        weighted = np.einsum("tij,tj->ti", self._flat[tilts], diff)
        loss += 0.5 * float((diff * weighted).sum())
        grad_shifts[tilts] += weighted
        grad_free = self._fix_gauge_grad(grad_offsets)
        grad_tilt = self._fix_tilt_gauge(grad_tilt, tilts) * self._tilt_unit
        return loss, np.concatenate(
            [grad_shifts.ravel(), [grad_angle], grad_tilt, grad_free.ravel()]
        )

    def _summed_map(
        self, corrections: _Corrections, t: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """The ZNCC (W, W) of the t-th tilt summed over the particles.

        The maps are shifted by the displacements of the particles by the current
        rotation and offsets before summation, so the summed ZNCC is of the shifts of
        the tilt image. Its center is at the returned shift (the center of the maps).
        """
        inner = (slice(_PAD, -_PAD),) * 2
        center = self._shift_centers[t]
        disp = self.displacements(corrections._replace(shifts=self.zeros().shifts), [t])
        # the map of each particle at the shifts relative to the center
        relative = disp[:, 0] + center - self._centers[:, t]
        total = np.zeros(self._raw.shape[2:])
        for n, (dy, dx) in enumerate(relative):
            total += ndi.shift(
                self._maps[n, t][inner],
                (-dy, -dx),
                order=1,
                mode="constant",
                cval=0.0,
            )
        return total, center

    def _consensus(self, corrections: _Corrections, t: int) -> NDArray[np.float64]:
        """The shift of the t-th tilt at the peak of the ZNCC summed over particles.

        The shifts within the radius from the center of the maps of the tilt are
        searched. The shift along the direction in which the summed ZNCC is flat, such
        as along parallel filaments, is not determined, so the shift at the center is
        kept along the direction, also by the prior of the optimization.
        """
        total, center = self._summed_map(corrections, t)
        if total.max() <= 0:
            self._flat[t] = 0.0
            return center.copy()
        peak, self._flat[t] = _split_directions(total, _subpixel_peak(total))
        return center + peak


def _split_directions(
    total: NDArray[np.float64], peak: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The peak along the curved directions of the map, and the prior of the others.

    The curvature is the Hessian of the smoothed map at the peak (relative to the
    center). The components of the peak along the eigenvectors whose curvature is less
    than ``_MIN_CURVATURE`` of the largest are removed, as the peak is not determined
    along them. The prior is the (2, 2) weight of the squared shift along the removed
    eigenvectors, as large as the largest curvature.
    """
    smoothed = ndi.gaussian_filter(total, 1.0, mode="nearest")
    radius = (total.shape[0] - 1) // 2
    iy, ix = np.clip(np.round(peak).astype(np.intp) + radius, 1, total.shape[0] - 2)
    s = smoothed[iy - 1 : iy + 2, ix - 1 : ix + 2]
    hyy = s[2, 1] - 2 * s[1, 1] + s[0, 1]
    hxx = s[1, 2] - 2 * s[1, 1] + s[1, 0]
    hxy = (s[2, 2] - s[2, 0] - s[0, 2] + s[0, 0]) / 4
    values, vectors = np.linalg.eigh(np.array([[hyy, hxy], [hxy, hxx]]))
    curvature = np.maximum(-values, 0.0)  # positive at a maximum
    if curvature.max() <= 0:
        return np.zeros(2), np.zeros((2, 2))
    keep = curvature >= _MIN_CURVATURE * curvature.max()
    curved, flat = vectors[:, keep], vectors[:, ~keep]
    return curved @ (curved.T @ peak), curvature.max() * flat @ flat.T


def _rotation_2d(angle: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Rotation matrix of (y, x) by ``angle`` in radian, and its derivative.

    Same as the rotation of the tilt images by ``tilt_axis_angle``.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, s], [-s, c]]), np.array([[-s, c], [-c, -s]])
