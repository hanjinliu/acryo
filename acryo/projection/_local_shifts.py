from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acryo._types import nm

if TYPE_CHECKING:
    from acryo.projection._tilt_series import TiltSeries

LocalShiftFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]

_ANGSTROM_PER_NM = 10.0
_CHUNK_SIZE = 4096  # number of points evaluated at once


class LocalShiftModel(ABC):
    """Base class of the models of the local shifts on the detector.

    A model is used as ``TiltSeries.local_shifts_2d``. It gives the shifts (nm) of the
    projected points, which may depend on the 3D positions of the points and the
    geometry of the tilt series. Models are serialized by :meth:`to_dict` and
    deserialized by :meth:`LocalShiftModel.from_dict`.
    """

    _registry: ClassVar[dict[str, type[LocalShiftModel]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        LocalShiftModel._registry[cls.__name__] = cls

    @abstractmethod
    def shifts(
        self,
        tilt_series: TiltSeries | None,
        points: NDArray[np.float64] | None,
        projected: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Shifts of the projected points.

        Parameters
        ----------
        tilt_series : TiltSeries
            The tilt series that projects the points.
        points : (N, 3) array
            (z, y, x) coordinates of the points in the sample space in nm.
        projected : (N, T, 2) array
            (y, x) coordinates of the projected points on the detector in nm, without
            the local shifts.

        Returns
        -------
        (N, T, 2) array
            (y, x) shifts in nm.
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""

    def linear_part(self) -> NDArray[np.float64] | None:
        """(T, 2, 2) linear transformation of the projected points of each tilt.

        If the model transforms the projected (y, x) coordinates of each tilt by a
        matrix everywhere on the detector, such as the change of magnification of
        :class:`AffineShifts2D`, the matrices are returned. The shifts are evaluated
        only at the centers of the patches of the tilt images in reconstruction, and
        the patches are transformed by these matrices. None if there is no such part.
        """
        return None

    @property
    def position_dependent(self) -> bool:
        """Whether the model is not a constant shift plus :meth:`linear_part`.

        If true, the patches of the tilt images around each point are reconstructed
        by the linear part of the projection around the point (see
        :meth:`TiltSeries.local_detector2tomo_at`). A model can be used only for the
        shifts of the points by returning False, such as :class:`LocalAlignment` of
        ``mode="shift"``.
        """
        return True

    def select_tilts(self, indices: ArrayLike) -> LocalShiftModel:
        """Return the model of the selected tilts."""
        raise NotImplementedError(f"{type(self).__name__} cannot select tilts.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LocalShiftModel:
        """Create a model from the dict made by :meth:`to_dict`."""
        name = data.get("type")
        model_type = LocalShiftModel._registry.get(str(name))
        if model_type is None:
            raise ValueError(f"Unknown type of local shift model: {name!r}.")
        return model_type._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> LocalShiftModel:
        raise TypeError(f"{cls.__name__} cannot be deserialized.")


def as_local_shift_model(obj: Any) -> LocalShiftModel | None:
    """Convert a function of the projected (N, T, 2) points to a local shift model."""
    if obj is None or isinstance(obj, LocalShiftModel):
        return obj
    if callable(obj):
        return FunctionShifts(obj)
    raise TypeError(f"Expected a LocalShiftModel or a callable, got {type(obj)}.")


class FunctionShifts(LocalShiftModel):
    """A function of the projected points as a local shift model.

    This model cannot be serialized.

    Parameters
    ----------
    func : callable
        Function that maps (N, T, 2) projected points (nm) to their shifts (nm).
    """

    def __init__(self, func: LocalShiftFunction):
        self.func = func

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.func!r})"

    def shifts(self, tilt_series, points, projected) -> NDArray[np.float64]:
        return self.func(projected)

    def to_dict(self) -> dict[str, Any]:
        raise TypeError(f"{self!r} cannot be serialized. Use a LocalShiftModel.")


@dataclass(frozen=True, eq=False)
class LocalShifts2D(LocalShiftModel):
    """Smooth 2D shifts of each tilt image interpolated from a grid of control points.

    The shifts are bilinearly interpolated by the projected positions, and constant
    outside the grid.

    Parameters
    ----------
    control_points : (T, 2, gy, gx) array
        (y, x) shifts in nm at the control points of each tilt image. The control
        points are evenly placed from edge to edge of the images.
    half_extent : (2,) array
        Half size of the tilt images (y, x) in nm.
    base : LocalShiftModel or callable, optional
        Local shifts applied before this model. The shifts of this model are
        evaluated at the projected positions shifted by the base.
    """

    control_points: NDArray[np.float64]
    half_extent: NDArray[np.float64]
    base: LocalShiftModel | None = None

    def __post_init__(self):
        control_points = np.asarray(self.control_points, dtype=np.float64)
        if control_points.ndim != 4 or control_points.shape[1] != 2:
            raise ValueError(
                "Shape of 'control_points' must be (T, 2, gy, gx), got "
                f"{control_points.shape}."
            )
        half_extent = np.asarray(self.half_extent, dtype=np.float64)
        if half_extent.shape != (2,):
            raise ValueError(
                f"Shape of 'half_extent' must be (2,), got {half_extent.shape}."
            )
        object.__setattr__(self, "control_points", _readonly(control_points))
        object.__setattr__(self, "half_extent", _readonly(half_extent))
        object.__setattr__(self, "base", as_local_shift_model(self.base))

    def __repr__(self) -> str:
        ntilts, _, gy, gx = self.control_points.shape
        return (
            f"{type(self).__name__}(grid_shape=({gy}, {gx}), ntilts={ntilts}, "
            f"base={self.base!r})"
        )

    def __call__(self, projected: NDArray[np.float64]) -> NDArray[np.float64]:
        """Local shifts of the projected (N, T, 2) points in nm.

        The base model is evaluated without the tilt series and the 3D points.
        """
        return self.shifts(None, None, projected)

    def shifts(self, tilt_series, points, projected) -> NDArray[np.float64]:
        base, projected = _apply_base(self, tilt_series, points, projected)
        grid = self.control_points
        ntilts, _, gy, gx = grid.shape
        normed = projected / self.half_extent
        # (N, T) continuous indices of the control points
        iy = np.clip((normed[..., 0] + 1) / 2 * (gy - 1), 0, gy - 1)
        ix = np.clip((normed[..., 1] + 1) / 2 * (gx - 1), 0, gx - 1)
        iy0 = np.minimum(np.floor(iy).astype(np.intp), max(gy - 2, 0))
        ix0 = np.minimum(np.floor(ix).astype(np.intp), max(gx - 2, 0))
        iy1 = np.minimum(iy0 + 1, gy - 1)
        ix1 = np.minimum(ix0 + 1, gx - 1)
        fy = (iy - iy0)[..., np.newaxis]
        fx = (ix - ix0)[..., np.newaxis]
        tilt = np.arange(ntilts)
        # (N, T, 2) values at the corners
        out = (
            grid[tilt, :, iy0, ix0] * (1 - fy) * (1 - fx)
            + grid[tilt, :, iy0, ix1] * (1 - fy) * fx
            + grid[tilt, :, iy1, ix0] * fy * (1 - fx)
            + grid[tilt, :, iy1, ix1] * fy * fx
        )
        return out if base is None else out + base

    def linear_part(self) -> NDArray[np.float64] | None:
        # the smooth shifts depend on the position
        return _linear_part(self.base)

    def select_tilts(self, indices: ArrayLike) -> LocalShifts2D:
        idx = np.asarray(indices, dtype=np.intp)
        return LocalShifts2D(
            control_points=self.control_points[idx],
            half_extent=self.half_extent,
            base=_select_tilts(self.base, idx),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "control_points": self.control_points.tolist(),
            "half_extent": self.half_extent.tolist(),
            "base": None if self.base is None else self.base.to_dict(),
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> LocalShifts2D:
        base = data.get("base")
        return cls(
            control_points=data["control_points"],
            half_extent=data["half_extent"],
            base=None if base is None else LocalShiftModel.from_dict(base),
        )


@dataclass(frozen=True, eq=False)
class AffineShifts2D(LocalShiftModel):
    """Affine transformation of each tilt image, such as a change of magnification.

    A projected point ``p`` of tilt ``t`` is moved to
    ``matrices[t] @ p + translations[t]``.

    Parameters
    ----------
    matrices : (T, 2, 2) array
        Linear parts of the transformations of the projected (y, x) coordinates
        relative to the center of the detector.
    translations : (T, 2) array, optional
        (y, x) translations in nm.
    base : LocalShiftModel or callable, optional
        Local shifts applied before this model. The shifts of this model are
        evaluated at the projected positions shifted by the base.
    """

    matrices: NDArray[np.float64]
    translations: NDArray[np.float64] = None  # type: ignore
    base: LocalShiftModel | None = None

    def __post_init__(self):
        matrices = np.asarray(self.matrices, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (2, 2):
            raise ValueError(
                f"Shape of 'matrices' must be (T, 2, 2), got {matrices.shape}."
            )
        ntilts = matrices.shape[0]
        if self.translations is None:
            translations = np.zeros((ntilts, 2))
        else:
            translations = np.asarray(self.translations, dtype=np.float64)
        if translations.shape != (ntilts, 2):
            raise ValueError(
                f"Shape of 'translations' must be ({ntilts}, 2), got "
                f"{translations.shape}."
            )
        object.__setattr__(self, "matrices", _readonly(matrices))
        object.__setattr__(self, "translations", _readonly(translations))
        object.__setattr__(self, "base", as_local_shift_model(self.base))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ntilts={self.matrices.shape[0]}, "
            f"base={self.base!r})"
        )

    def shifts(self, tilt_series, points, projected) -> NDArray[np.float64]:
        base, projected = _apply_base(self, tilt_series, points, projected)
        linear = self.matrices - np.eye(2)
        out = linear[:, :, 0] * projected[..., 0:1]
        out += linear[:, :, 1] * projected[..., 1:2] + self.translations
        return out if base is None else out + base

    def linear_part(self) -> NDArray[np.float64] | None:
        # the base is applied before this model
        base = _linear_part(self.base)
        return self.matrices if base is None else self.matrices @ base

    @property
    def position_dependent(self) -> bool:
        return self.base is not None and self.base.position_dependent

    def select_tilts(self, indices: ArrayLike) -> AffineShifts2D:
        idx = np.asarray(indices, dtype=np.intp)
        return AffineShifts2D(
            matrices=self.matrices[idx],
            translations=self.translations[idx],
            base=_select_tilts(self.base, idx),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "matrices": self.matrices.tolist(),
            "translations": self.translations.tolist(),
            "base": None if self.base is None else self.base.to_dict(),
        }

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> AffineShifts2D:
        base = data.get("base")
        return cls(
            matrices=data["matrices"],
            translations=data.get("translations"),
            base=None if base is None else LocalShiftModel.from_dict(base),
        )


@dataclass(frozen=True, eq=False)
class LocalAlignment(LocalShiftModel):
    """Local alignments of areas of the sample, such as the local alignments of IMOD.

    The sample is divided into a grid of local areas in its (y, x) plane. Each local
    area has increments of the alignment parameters of each tilt to those of the tilt
    series: the tilt angle, the X-axis tilt angle, the Z factors and a 2D affine
    refinement. In each local area, a point ``p`` in the sample space is projected
    (before the rotation by the tilt axis angle) by

    .. code-block:: python

        q = matrix @ (sample2scope(tilt_angle + d_tilt, x_tilt + d_x_tilt) @ p)[y, x]
        q += z_factor * p[z] + translation

    The projected positions of the four nearest local areas are bilinearly
    interpolated by the (y, x) position of the point (the local areas at the edges are
    used outside the grid). The shift is the difference from the projection of the
    tilt series, rotated by the tilt axis angle of the tilt series.

    Parameters
    ----------
    origin : (2,) array
        (y, x) coordinates (nm) of the center of the first local area in the sample
        space.
    spacing : (2,) array
        (y, x) spacing (nm) between the centers of the local areas.
    tilt_angles : (gy, gx, T) array
        Increments of the tilt angles in degrees.
    x_tilts : (gy, gx, T) array, optional
        Increments of the X-axis tilt angles in degrees.
    z_factors : (gy, gx, T, 2) array, optional
        (y, x) displacements per unit z (z coordinate in the sample space).
    matrices : (gy, gx, T, 2, 2) array, optional
        Linear parts of the refinements of the projected (y, x) coordinates. Identity
        by default.
    translations : (gy, gx, T, 2) array, optional
        (y, x) translations (nm) of the refinements.
    base : LocalShiftModel or callable, optional
        Local shifts applied before this model. The shifts of this model are
        evaluated at the projected positions shifted by the base.
    mode : {"shift", "full"}, default is "shift"
        How the local alignments are used. "shift" uses them only for the shifts of
        the projected points, and the patches of the tilt images around them are
        reconstructed (and the templates are projected) by the geometry of the whole
        tilt series (:attr:`TiltSeries.local_detector2tomo`). "full" also uses the
        local tilt angles, rotation and magnification around each point
        (:meth:`TiltSeries.local_detector2tomo_at`, see :attr:`position_dependent`).
    """

    origin: NDArray[np.float64]
    spacing: NDArray[np.float64]
    tilt_angles: NDArray[np.float64]
    x_tilts: NDArray[np.float64] = None  # type: ignore
    z_factors: NDArray[np.float64] = None  # type: ignore
    matrices: NDArray[np.float64] = None  # type: ignore
    translations: NDArray[np.float64] = None  # type: ignore
    base: LocalShiftModel | None = None
    mode: Literal["shift", "full"] = "shift"

    def __post_init__(self):
        _set = object.__setattr__
        if self.mode not in ("shift", "full"):
            raise ValueError(f"'mode' must be 'shift' or 'full', got {self.mode!r}.")
        tilt_angles = np.asarray(self.tilt_angles, dtype=np.float64)
        if tilt_angles.ndim != 3:
            raise ValueError(
                f"Shape of 'tilt_angles' must be (gy, gx, T), got {tilt_angles.shape}."
            )
        _set(self, "tilt_angles", _readonly(tilt_angles))
        shape = tilt_angles.shape
        defaults = {
            "x_tilts": (shape, 0.0),
            "z_factors": (shape + (2,), 0.0),
            "matrices": (shape + (2, 2), np.eye(2)),
            "translations": (shape + (2,), 0.0),
        }
        for name, (expected, default) in defaults.items():
            value = getattr(self, name)
            if value is None:
                arr = np.broadcast_to(default, expected)
            else:
                arr = np.asarray(value, dtype=np.float64)
            if arr.shape != expected:
                raise ValueError(
                    f"Shape of '{name}' must be {expected}, got {arr.shape}."
                )
            _set(self, name, _readonly(arr))
        for name in ["origin", "spacing"]:
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.shape != (2,):
                raise ValueError(f"Shape of '{name}' must be (2,), got {arr.shape}.")
            _set(self, name, _readonly(arr))
        if np.any(self.spacing <= 0):
            raise ValueError(f"'spacing' must be positive, got {self.spacing}.")
        _set(self, "base", as_local_shift_model(self.base))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(grid_shape={self.grid_shape}, "
            f"ntilts={self.ntilts}, origin={self.origin.tolist()}, "
            f"spacing={self.spacing.tolist()}, mode={self.mode!r}, "
            f"base={self.base!r})"
        )

    @property
    def grid_shape(self) -> tuple[int, int]:
        """(gy, gx) number of the local areas."""
        gy, gx, _ = self.tilt_angles.shape
        return gy, gx

    @property
    def ntilts(self) -> int:
        """Number of tilts."""
        return self.tilt_angles.shape[2]

    def shifts(self, tilt_series, points, projected) -> NDArray[np.float64]:
        if tilt_series is None or points is None:
            raise ValueError(
                f"{type(self).__name__} needs the tilt series and the 3D points."
            )
        if tilt_series.ntilts != self.ntilts:
            raise ValueError(
                f"{type(self).__name__} has {self.ntilts} tilts, but the tilt series "
                f"has {tilt_series.ntilts} tilts."
            )
        base, _ = _apply_base(self, tilt_series, points, projected)
        # (gy, gx, T, 2, 3) projections of each local area, including the refinement
        # and the Z factors, and that of the tilt series
        local = self.matrices @ _scope_matrices(
            tilt_series.tilt_angles + self.tilt_angles,
            tilt_series.x_tilts + self.x_tilts,
        )
        local[..., 0] += self.z_factors
        glob = _scope_matrices(tilt_series.tilt_angles, tilt_series.x_tilts)
        delta = np.empty(projected.shape, dtype=np.float64)
        for start in range(0, points.shape[0], _CHUNK_SIZE):
            sl = slice(start, start + _CHUNK_SIZE)
            chunk = points[sl]
            linear = -glob
            offset = 0.0
            for iy, ix, weight in self._corners(chunk[:, 1:3]):
                weight = weight[:, np.newaxis, np.newaxis]
                linear = linear + weight[..., np.newaxis] * local[iy, ix]
                offset = offset + weight * self.translations[iy, ix]
            delta[sl] = (linear * chunk[:, np.newaxis, np.newaxis]).sum(axis=-1)
            delta[sl] += offset
        # rotation by the tilt axis angle, same as `TiltSeries.scope2detector`
        rad = np.deg2rad(tilt_series.tilt_axis_angle)
        c, s = np.cos(rad), np.sin(rad)
        dy, dx = delta[..., 0], delta[..., 1]
        out = np.stack([c * dy + s * dx, -s * dy + c * dx], axis=-1)
        return out if base is None else out + base

    def linear_part(self) -> NDArray[np.float64] | None:
        # the alignments of the local areas depend on the position
        return _linear_part(self.base)

    @property
    def position_dependent(self) -> bool:
        # the local geometry is used only in the "full" mode
        if self.mode == "full":
            return True
        return self.base is not None and self.base.position_dependent

    def _corners(self, yx: NDArray[np.float64]):
        """Indices and bilinear weights of the four nearest local areas."""
        axes = []
        for axis, size in enumerate(self.grid_shape):
            pos = (yx[:, axis] - self.origin[axis]) / self.spacing[axis]
            pos = np.clip(pos, 0, size - 1)
            i0 = np.minimum(np.floor(pos).astype(np.intp), max(size - 2, 0))
            i1 = np.minimum(i0 + 1, size - 1)
            axes.append((i0, i1, pos - i0))
        (y0, y1, fy), (x0, x1, fx) = axes
        return [
            (y0, x0, (1 - fy) * (1 - fx)),
            (y0, x1, (1 - fy) * fx),
            (y1, x0, fy * (1 - fx)),
            (y1, x1, fy * fx),
        ]

    def select_tilts(self, indices: ArrayLike) -> LocalAlignment:
        idx = np.asarray(indices, dtype=np.intp)
        return LocalAlignment(
            origin=self.origin,
            spacing=self.spacing,
            tilt_angles=self.tilt_angles[:, :, idx],
            x_tilts=self.x_tilts[:, :, idx],
            z_factors=self.z_factors[:, :, idx],
            matrices=self.matrices[:, :, idx],
            translations=self.translations[:, :, idx],
            base=_select_tilts(self.base, idx),
            mode=self.mode,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": type(self).__name__,
            "origin": self.origin.tolist(),
            "spacing": self.spacing.tolist(),
            "tilt_angles": self.tilt_angles.tolist(),
        }
        # parameters of the default values are omitted
        if np.any(self.x_tilts != 0):
            out["x_tilts"] = self.x_tilts.tolist()
        if np.any(self.z_factors != 0):
            out["z_factors"] = self.z_factors.tolist()
        if np.any(self.matrices != np.eye(2)):
            out["matrices"] = self.matrices.tolist()
        if np.any(self.translations != 0):
            out["translations"] = self.translations.tolist()
        out["base"] = None if self.base is None else self.base.to_dict()
        out["mode"] = self.mode
        return out

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> LocalAlignment:
        return cls(
            origin=data["origin"],
            spacing=data["spacing"],
            tilt_angles=data["tilt_angles"],
            x_tilts=data.get("x_tilts"),
            z_factors=data.get("z_factors"),
            matrices=data.get("matrices"),
            translations=data.get("translations"),
            base=None if data.get("base") is None else cls.from_dict(data["base"]),
            mode=data.get("mode", "shift"),
        )

    @classmethod
    def from_imod(
        cls,
        path: str | Path,
        image_shape: tuple[int, int],
        scale: nm,
        *,
        z_factors: bool = True,
        mode: Literal["shift", "full"] = "shift",
    ) -> LocalAlignment:
        """Read the local alignments of IMOD (``OutputLocalFile`` of tiltalign).

        The local alignments are converted for the tilt series made by
        :meth:`TiltSeries.from_etomo_directory`, whose sample space is centered at the
        center of the tilt images.

        Parameters
        ----------
        path : str or Path
            The local alignment file, such as ``*local.xf``.
        image_shape : (int, int)
            (H, W) shape of the tilt images (``FULLIMAGE`` of tilt.com) of the pixel
            size ``scale``.
        scale : float
            Pixel size of the tilt images in nm.
        z_factors : bool, default is True
            Whether to read the Z factors. ``tilt`` of IMOD ignores them unless
            ``ZFACTORFILE`` is given.
        mode : {"shift", "full"}, default is "shift"
            How the local alignments are used (see :class:`LocalAlignment`).
        """
        path = Path(path)
        header, *lines = path.read_text().splitlines()
        tokens = header.split()
        if len(tokens) < 7:
            raise ValueError(f"{path} is not a local alignment file of IMOD.")
        nx, ny = int(tokens[0]), int(tokens[1])
        x_start, y_start, dx, dy = (float(t) for t in tokens[2:6])
        if_xtilts = int(tokens[6])
        # pixel size (Å) and Z factors are absent in the files of old versions
        pixel = float(tokens[7]) / _ANGSTROM_PER_NM if len(tokens) > 7 else 0.0
        if_zfactors = int(tokens[8]) if len(tokens) > 8 else 0
        if pixel <= 0:
            pixel = scale
        values = np.array(" ".join(lines).split(), dtype=np.float64)
        per_tilt = 1 + if_xtilts + 2 * if_zfactors + 6
        ntilts, rest = divmod(values.size, nx * ny * per_tilt)
        if rest != 0 or ntilts == 0:
            raise ValueError(
                f"{path} does not contain the local alignments of {nx} x {ny} areas."
            )
        areas = values.reshape(ny, nx, per_tilt * ntilts)
        tilt_angles = areas[..., :ntilts]
        i = ntilts
        x_tilts = None
        if if_xtilts:
            x_tilts = areas[..., i : i + ntilts]
            i += ntilts
        zfac = None
        if if_zfactors:
            if z_factors:
                # pairs of (x, y) -> (y, x)
                zfac = areas[..., i : i + 2 * ntilts].reshape(ny, nx, ntilts, 2)
                zfac = zfac[..., ::-1]
            i += 2 * ntilts
        # A11 A12 A21 A22 DX DY of (x, y) -> matrices and translations of (y, x)
        xf = areas[..., i:].reshape(ny, nx, ntilts, 6)
        matrices = np.stack(
            [
                np.stack([xf[..., 3], xf[..., 2]], axis=-1),
                np.stack([xf[..., 1], xf[..., 0]], axis=-1),
            ],
            axis=-2,
        )
        translations = xf[..., [5, 4]] * pixel
        center = np.asarray(image_shape, dtype=np.float64) / 2 * scale
        return cls(
            origin=np.array([y_start, x_start]) * pixel - center,
            spacing=np.array([dy, dx]) * pixel,
            tilt_angles=tilt_angles,
            x_tilts=x_tilts,
            z_factors=zfac,
            matrices=matrices,
            translations=translations,
            mode=mode,
        )


def _apply_base(
    model, tilt_series, points, projected: NDArray[np.float64]
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64]]:
    """Shifts of the base of the model, and the projected points shifted by them."""
    if model.base is None:
        return None, projected
    base = model.base.shifts(tilt_series, points, projected)
    return base, projected + base


def _select_tilts(model: LocalShiftModel | None, indices) -> LocalShiftModel | None:
    return None if model is None else model.select_tilts(indices)


def _linear_part(model: LocalShiftModel | None) -> NDArray[np.float64] | None:
    return None if model is None else model.linear_part()


def _scope_matrices(theta: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
    """(..., 2, 3) matrices from the sample (z, y, x) to the projected (y, x).

    Same as ``TiltSeries.sample2scope``: rotation by ``alpha`` around the x axis
    followed by the rotation by ``theta`` around the y axis (degrees). The tilt axis
    is not rotated yet.
    """
    t, a = np.deg2rad(theta), np.deg2rad(alpha)
    ct, st, ca, sa = np.broadcast_arrays(np.cos(t), np.sin(t), np.cos(a), np.sin(a))
    row_y = np.stack([-sa, ca, np.zeros_like(ca)], axis=-1)
    row_x = np.stack([st * ca, st * sa, ct], axis=-1)
    return np.stack([row_y, row_x], axis=-2)


def _readonly(arr: ArrayLike) -> NDArray:
    arr = np.array(arr)  # copy
    arr.setflags(write=False)
    return arr
