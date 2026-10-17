from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
import json
from typing import TYPE_CHECKING, Any, Callable, Sequence
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acryo._types import nm

if TYPE_CHECKING:
    from typing_extensions import Self
    from torch_tilt_series import TiltSeries as TorchTiltSeries

LocalShiftFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]

_ANGSTROM_PER_NM = 10.0


@dataclass(frozen=True, eq=False)
class TiltSeries:
    """Alignment geometry of a tilt series, and the projection of 3D points.

    This is a numpy port of ``torch_tilt_series.TiltSeries``, but all the lengths
    are in nm instead of Å. Coordinates are in (z, y, x) order. Points in the
    tomogram space are projected to the detector by

    1. ``tomo2sample`` (tomogram space -> sample space),
    2. ``local_shifts`` (deformation of the sample, optional),
    3. ``sample2scope`` (rotation by ``x_tilts`` around the x axis, followed by the
       rotation by ``tilt_angles`` around the y axis),
    4. ``scope2detector`` (rotation by ``tilt_axis_angle`` around the optical axis,
       followed by the translation by ``sample_translations``),
    5. ``local_shifts_2d`` (shifts on the detector, optional).

    Instances are immutable. Use :meth:`replace` to create a modified one.

    Parameters
    ----------
    tilt_angles : (T,) array
        Tilt angles in degrees.
    tilt_axis_angle : float or (T,) array
        Angle of the tilt axis on the detector in degrees.
    sample_translations : (T, 2) array, optional
        (y, x) translations of the projections on the detector in nm.
    x_tilts : float or (T,) array, default is 0.0
        Tilt of the sample around the x axis in degrees.
    sample2levelled, levelled2tomo : (4, 4) array, optional
        Affine matrices (translation in nm) from the sample space to the levelled
        sample space, and from the levelled sample space to the tomogram space.
    local_shifts : callable, optional
        Function that maps (N, 3) points in the sample space (nm) to their shifts.
    local_shifts_2d : callable, optional
        Function that maps (N, T, 2) projected points (nm) to their shifts.
    image_path : str, Path or sequence of them, optional
        Path to the MRC file of the stack of the tilt images, or the paths to the MRC
        files of each tilt image (such as the motion-corrected tilt images of RELION).
    image_indices : (T,) array of int, optional
        Indices of the images in the stack corresponding to the tilts. If
        ``image_path`` is a sequence, indices of the images in each file (0 by
        default).
    scale : float, optional
        Pixel size of the tilt images in nm.
    """

    tilt_angles: NDArray[np.float64]
    tilt_axis_angle: NDArray[np.float64]
    sample_translations: NDArray[np.float64] = None  # type: ignore
    x_tilts: NDArray[np.float64] = 0.0  # type: ignore
    sample2levelled: NDArray[np.float64] = None  # type: ignore
    levelled2tomo: NDArray[np.float64] = None  # type: ignore
    local_shifts: LocalShiftFunction | None = None
    local_shifts_2d: LocalShiftFunction | None = None
    image_path: Path | tuple[Path, ...] | None = None
    image_indices: NDArray[np.intp] | None = None
    scale: nm | None = None

    def __post_init__(self):
        tilt_angles = _readonly(np.asarray(self.tilt_angles, dtype=np.float64))
        if tilt_angles.ndim != 1 or tilt_angles.size == 0:
            raise ValueError(
                f"'tilt_angles' must be a non-empty 1D array, got {tilt_angles.shape}."
            )
        ntilts = tilt_angles.size
        _set = object.__setattr__
        _set(self, "tilt_angles", tilt_angles)
        _set(
            self,
            "tilt_axis_angle",
            _per_tilt(self.tilt_axis_angle, ntilts, "tilt_axis_angle"),
        )
        _set(self, "x_tilts", _per_tilt(self.x_tilts, ntilts, "x_tilts"))
        if self.sample_translations is None:
            translations = np.zeros((ntilts, 2))
        else:
            translations = np.asarray(self.sample_translations, dtype=np.float64)
        if translations.shape != (ntilts, 2):
            raise ValueError(
                f"Shape of 'sample_translations' must be ({ntilts}, 2), got "
                f"{translations.shape}."
            )
        _set(self, "sample_translations", _readonly(translations))
        for name in ["sample2levelled", "levelled2tomo"]:
            value = getattr(self, name)
            mtx = np.eye(4) if value is None else np.asarray(value, dtype=np.float64)
            if mtx.shape != (4, 4):
                raise ValueError(f"Shape of '{name}' must be (4, 4), got {mtx.shape}.")
            _set(self, name, _readonly(mtx))
        if self.image_path is not None:
            _set(self, "image_path", _normalize_image_path(self.image_path, ntilts))
        if self.image_indices is not None:
            indices = np.asarray(self.image_indices)
            if indices.shape != (ntilts,) or not np.issubdtype(
                indices.dtype, np.integer
            ):
                raise ValueError(
                    f"'image_indices' must be ({ntilts},) integers, got {indices.shape} "
                    f"{indices.dtype}."
                )
            _set(self, "image_indices", _readonly(indices.astype(np.intp)))
        if self.scale is not None:
            if self.scale <= 0:
                raise ValueError(f"'scale' must be positive, got {self.scale}.")
            _set(self, "scale", float(self.scale))

    def __repr__(self) -> str:
        angles = self.tilt_angles
        if isinstance(self.image_path, tuple):
            image_path = (
                f"<{len(self.image_path)} files in {self.image_path[0].parent}>"
            )
        else:
            image_path = str(self.image_path)
        return (
            f"{type(self).__name__}(ntilts={self.ntilts}, tilt_range=({angles.min():.1f}, "
            f"{angles.max():.1f}), scale={self.scale}, image_path={image_path})"
        )

    @property
    def ntilts(self) -> int:
        """Number of tilts."""
        return self.tilt_angles.size

    def replace(self, **kwargs: Any) -> Self:
        """Return a new tilt series with the given attributes replaced."""
        return dataclasses.replace(self, **kwargs)

    @property
    def sample2tomo(self) -> NDArray[np.float64]:
        """(4, 4) affine matrix from the sample space to the tomogram space."""
        return self.levelled2tomo @ self.sample2levelled

    @property
    def tomo2sample(self) -> NDArray[np.float64]:
        """(4, 4) affine matrix from the tomogram space to the sample space."""
        return np.linalg.inv(self.sample2tomo)

    @property
    def sample2scope(self) -> NDArray[np.float64]:
        """(T, 4, 4) rotations from the sample space to the microscope space."""
        return _rotation(self.tilt_angles, "y") @ _rotation(self.x_tilts, "x")

    @property
    def scope2detector(self) -> NDArray[np.float64]:
        """(T, 4, 4) affine matrices from the microscope space to the detector."""
        shifts = np.zeros((self.ntilts, 3))
        shifts[:, 1:] = self.sample_translations
        return _translation(shifts) @ _rotation(self.tilt_axis_angle, "z")

    @property
    def projection_matrices(self) -> NDArray[np.float64]:
        """(T, 4, 4) affine matrices from the sample space to the detector."""
        return self.scope2detector @ self.sample2scope

    @property
    def detector2tomo(self) -> NDArray[np.float64]:
        """(T, 3, 3) rotations from the detector (z, y, x) to the tomogram space.

        The image plane of each tilt (z = 0) is mapped to the central slice of the
        tomogram space, which is used for the reconstruction.
        """
        linear = self.projection_matrices[:, :3, :3] @ self.tomo2sample[:3, :3]
        return np.linalg.pinv(linear)

    def project_points(self, points: ArrayLike) -> NDArray[np.float64]:
        """Project points in the tomogram space to the detector.

        Parameters
        ----------
        points : (..., 3) array
            (z, y, x) coordinates in nm relative to the center of the tomogram.

        Returns
        -------
        (..., T, 2) array
            (y, x) coordinates in nm relative to the center of the detector.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1:] != (3,):
            raise ValueError(f"Shape of points must be (..., 3), got {points.shape}.")
        batch_shape = points.shape[:-1]
        points = points.reshape(-1, 3)
        points = _homogenize(points) @ self.tomo2sample.T
        points = points[:, :3]
        if self.local_shifts is not None:
            points = points + self.local_shifts(points)
        matrices = self.projection_matrices[:, 1:3, :]  # (T, 2, 4)
        projected = np.einsum("tij,nj->nti", matrices, _homogenize(points))
        if self.local_shifts_2d is not None:
            projected = projected + self.local_shifts_2d(projected)
        return projected.reshape(batch_shape + projected.shape[1:])

    @classmethod
    def from_torch(cls, tilt_series: TorchTiltSeries) -> TiltSeries:
        """Convert a ``torch_tilt_series.TiltSeries`` (in Å) to a tilt series."""

        def _numpy(tensor) -> NDArray[np.float64]:
            return tensor.detach().cpu().numpy().astype(np.float64)

        if (indices := tilt_series.image_indices) is not None:
            indices = indices.cpu().numpy()
        spacing = tilt_series._pixel_spacing
        return cls(
            tilt_angles=_numpy(tilt_series.tilt_angles),
            tilt_axis_angle=_numpy(tilt_series.tilt_axis_angle),
            sample_translations=_numpy(tilt_series.sample_translations)
            / _ANGSTROM_PER_NM,
            x_tilts=_numpy(tilt_series.x_tilts),
            sample2levelled=_affine_to_nm(_numpy(tilt_series.sample2levelled)),
            levelled2tomo=_affine_to_nm(_numpy(tilt_series.levelled2tomo)),
            local_shifts=_shift_function_from_torch(tilt_series.local_shifts),
            local_shifts_2d=_shift_function_from_torch(tilt_series.local_shifts_2d),
            image_path=tilt_series.image_path,
            image_indices=indices,
            scale=None if spacing is None else spacing / _ANGSTROM_PER_NM,
        )

    def to_torch(self, device="cpu") -> TorchTiltSeries:
        """Convert to a ``torch_tilt_series.TiltSeries`` (in Å).

        If the tilt images are in separate files, ``image_path`` is not converted,
        because ``torch_tilt_series.TiltSeries`` only supports a stack.
        """
        from torch_tilt_series import TiltSeries as TorchTiltSeries

        if isinstance(self.image_path, tuple):
            image_path, image_indices = None, None
        else:
            image_path, image_indices = self.image_path, self.image_indices
        return TorchTiltSeries(
            tilt_angles=self.tilt_angles,
            tilt_axis_angle=self.tilt_axis_angle,
            sample_translations=self.sample_translations * _ANGSTROM_PER_NM,
            x_tilts=self.x_tilts,
            sample2levelled=_affine_to_angstrom(self.sample2levelled),
            levelled2tomo=_affine_to_angstrom(self.levelled2tomo),
            local_shifts=_shift_function_to_torch(self.local_shifts),
            local_shifts_2d=_shift_function_to_torch(self.local_shifts_2d),
            image_path=image_path,
            image_indices=image_indices,
            pixel_spacing=None if self.scale is None else self.scale * _ANGSTROM_PER_NM,
            device=device,
        )

    @classmethod
    def from_etomo_directory(cls, path: str | Path, scale: nm) -> TiltSeries:
        """Read the alignment of an IMOD etomo directory.

        ``etomofiles`` is required. Excluded views are dropped.

        Parameters
        ----------
        path : str or Path
            The etomo directory.
        scale : float
            Pixel size of the raw tilt images in nm.
        """
        import etomofiles

        path = Path(path)
        df = etomofiles.read(path)
        df = df.loc[~df["excluded"]].reset_index(drop=True)
        image_path = Path(df["image_path"][0].split("[")[0])
        if not image_path.is_absolute():
            image_path = path / image_path
        xf = etomofiles.df_to_xf(df, yx=True)  # (T, 2, 3) in (y, x)
        mtx, shifts = xf[:, :, :2], xf[:, :, 2]
        # IMOD transforms the images to the aligned ones, which is the inverse of
        # the projection model. The rotation is orthogonal, so it is the transpose.
        translations = -np.einsum("nji,nj->ni", mtx, shifts) * scale
        # The sign of IMOD's XAXISTILT is opposite.
        xaxistilt = -(df["xaxistilt"].iloc[0] or 0.0)
        x_tilts = xaxistilt + np.nan_to_num(df["xtilt"].to_numpy(), nan=0.0)
        return cls(
            tilt_angles=df["tlt"].to_numpy(),
            tilt_axis_angle=df["tilt_axis_angle"].to_numpy(),
            sample_translations=translations,
            x_tilts=x_tilts,
            image_path=image_path,
            image_indices=df["idx_tilt"].to_numpy(),
            scale=scale,
        )

    @classmethod
    def from_aretomo_output(
        cls,
        path: str | Path,
        scale: nm,
        image_path: str | Path | None = None,
    ) -> TiltSeries:
        """Read the alignment of an AreTomo .aln file.

        ``alnfile`` is required.

        Parameters
        ----------
        path : str or Path
            The .aln file.
        scale : float
            Pixel size of the raw tilt images in nm.
        image_path : str or Path, optional
            Path to the tilt images. By default, the .mrc file of the same name.
        """
        import alnfile

        path = Path(path)
        df = alnfile.read(path)
        if image_path is None:
            image_path = path.with_suffix(".mrc")
        translations = df[["ty", "tx"]].to_numpy(dtype=np.float64) * scale
        return cls(
            tilt_angles=df["tilt"].to_numpy(),
            tilt_axis_angle=df["rot"].to_numpy(),
            sample_translations=translations,
            image_path=image_path,
            image_indices=df["sec"].to_numpy() - 1,
            scale=scale,
        )

    @classmethod
    def from_relion_tilt_series_star(
        cls,
        path: str | Path,
        tomo_name: str | None = None,
        *,
        project_dir: str | Path | None = None,
        scale: nm | None = None,
    ) -> TiltSeries:
        """Read a tilt series of RELION 5.

        The alignment (``rlnTomoXTilt``, ``rlnTomoYTilt``, ``rlnTomoZRot``,
        ``rlnTomoXShiftAngst`` and ``rlnTomoYShiftAngst``) and the motion-corrected
        tilt images (``rlnMicrographName``) of each tilt are read. RELION projects a
        point by the rotations around the x, y and z axes in this order, followed by
        the shift, which is the same as this class. The tomogram space is centered,
        so the centered coordinates of RELION (such as
        ``rlnCenteredCoordinateXAngst``) are the coordinates of this class. Tilt
        series that are not aligned yet are read with the nominal tilt angles and
        tilt axis angle. ``starfile-rs`` is required.

        Parameters
        ----------
        path : str or Path
            STAR file of tilt series with the "global" data block that lists the tilt
            series (such as ``tilt_series.star`` or ``tomograms.star``), or the STAR
            file of a single tilt series (such as ``tilt_series/TS_01.star``).
        tomo_name : str, optional
            Name of the tilt series (``rlnTomoName``) to read. It is required if the
            STAR file lists more than one tilt series.
        project_dir : str or Path, optional
            The RELION project directory, to which the paths in the STAR files are
            relative. By default, the nearest parent directory of the STAR file where
            the paths are found.
        scale : float, optional
            Pixel size of the tilt images in nm. By default,
            ``rlnTomoTiltSeriesPixelSize`` of the global data block, or the pixel
            size in the header of the first tilt image.
        """
        from starfile_rs import read_star

        path = Path(path)
        star = read_star(path)
        if (block := star.get("global")) is not None and (
            "rlnTomoTiltSeriesStarFile" in block.columns
        ):
            series = _select_relion_tilt_series(block.to_polars(), tomo_name, path)
            ts_path = Path(series["rlnTomoTiltSeriesStarFile"])
            project_dir = _relion_project_dir(path, ts_path, project_dir)
            if not ts_path.is_absolute():
                ts_path = project_dir / ts_path
            if scale is None and series.get("rlnTomoTiltSeriesPixelSize") is not None:
                scale = float(series["rlnTomoTiltSeriesPixelSize"]) / _ANGSTROM_PER_NM
            tomo_name = str(series["rlnTomoName"])
            star = read_star(ts_path)
        df = _select_relion_tilt_block(star, tomo_name, path).to_polars()

        image_paths: list[Path] = []
        image_indices: list[int] = []
        for name in df["rlnMicrographName"].cast(str).to_list():
            if "@" in name:
                # "N@file" is the N-th (1-based) image in the file
                index, name = name.split("@", 1)
                image_indices.append(int(index) - 1)
            else:
                image_indices.append(0)
            image_paths.append(Path(name))
        project_dir = _relion_project_dir(path, image_paths[0], project_dir)
        image_paths = [p if p.is_absolute() else project_dir / p for p in image_paths]

        columns = df.columns
        if all(c in columns for c in _RELION_ALIGNMENT_COLUMNS):
            x_tilts = df["rlnTomoXTilt"].to_numpy()
            tilt_angles = df["rlnTomoYTilt"].to_numpy()
            tilt_axis_angle = df["rlnTomoZRot"].to_numpy()
            shifts_xy = df[["rlnTomoXShiftAngst", "rlnTomoYShiftAngst"]].to_numpy()
            # (y, x) in nm
            translations = shifts_xy[:, ::-1].astype(np.float64) / _ANGSTROM_PER_NM
        else:
            warnings.warn(
                f"The tilt series in {path} is not aligned. The nominal tilt angles "
                "and tilt axis angle are used.",
                UserWarning,
                stacklevel=2,
            )
            x_tilts = 0.0
            tilt_angles = df["rlnTomoNominalStageTiltAngle"].to_numpy()
            tilt_axis_angle = df["rlnTomoNominalTiltAxisAngle"].to_numpy()
            translations = np.zeros((len(df), 2))

        first_image = image_paths[0]
        if first_image.exists():
            from acryo.projection._io import MrcImageStack

            stack = MrcImageStack(first_image)
            if scale is None and stack.voxel_size > 0:
                scale = stack.voxel_size / _ANGSTROM_PER_NM
            if scale is not None:
                # The origin of the detector is at size / 2 (pixel) of the images in
                # RELION, but at size // 2 in this class.
                ny, nx = stack.shape[1:]
                translations = translations + np.array([ny % 2, nx % 2]) * scale / 2
        if scale is None:
            raise ValueError(
                f"The pixel size of the tilt images is not found in {path}. Give "
                "'scale' explicitly."
            )
        return cls(
            tilt_angles=tilt_angles,
            tilt_axis_angle=tilt_axis_angle,
            sample_translations=translations,
            x_tilts=x_tilts,
            image_path=tuple(image_paths),
            image_indices=None if not any(image_indices) else image_indices,
            scale=scale,
        )

    @classmethod
    def from_json(cls, json_str: str) -> TiltSeries:
        data = json.loads(json_str)
        image_path = data["image_path"]
        if isinstance(image_path, list):
            image_path = tuple(image_path)
        image_indices = (
            np.array(data["image_indices"], dtype=np.int64)
            if data["image_indices"] is not None
            else None
        )
        # the geometry below may be missing in the json files of older versions
        sample2levelled = data.get("sample2levelled")
        levelled2tomo = data.get("levelled2tomo")
        return cls(
            tilt_angles=np.array(data["tilt_angles"], dtype=np.float64),
            tilt_axis_angle=np.array(data["tilt_axis_angle"], dtype=np.float64),
            sample_translations=np.array(data["sample_translations"], dtype=np.float64),
            x_tilts=np.array(data.get("x_tilts", 0.0), dtype=np.float64),
            sample2levelled=(
                np.array(sample2levelled, dtype=np.float64)
                if sample2levelled is not None
                else None
            ),
            levelled2tomo=(
                np.array(levelled2tomo, dtype=np.float64)
                if levelled2tomo is not None
                else None
            ),
            image_path=image_path,
            image_indices=image_indices,
            scale=data["scale"],
        )

    def dump_json(self) -> str:
        if isinstance(self.image_path, tuple):
            image_path: str | list[str] | None = [str(p) for p in self.image_path]
        elif self.image_path is not None:
            image_path = str(self.image_path)
        else:
            image_path = None
        image_indices = (
            self.image_indices.tolist() if self.image_indices is not None else None
        )
        return json.dumps(
            {
                "tilt_angles": self.tilt_angles.tolist(),
                "tilt_axis_angle": self.tilt_axis_angle.tolist(),
                "sample_translations": self.sample_translations.tolist(),
                "x_tilts": self.x_tilts.tolist(),
                "sample2levelled": self.sample2levelled.tolist(),
                "levelled2tomo": self.levelled2tomo.tolist(),
                "image_path": image_path,
                "image_indices": image_indices,
                "scale": self.scale,
            }
        )


@dataclass(frozen=True, eq=False)
class LocalShifts2D:
    """Smooth 2D shifts of each tilt image interpolated from a grid of control points.

    This object can be used as ``TiltSeries.local_shifts_2d``. The shifts are
    bilinearly interpolated, and constant outside the grid.

    Parameters
    ----------
    control_points : (T, 2, gy, gx) array
        (y, x) shifts in nm at the control points of each tilt image. The control
        points are evenly placed from edge to edge of the images.
    half_extent : (2,) array
        Half size of the tilt images (y, x) in nm.
    base : callable, optional
        Local shifts to be added.
    """

    control_points: NDArray[np.float64]
    half_extent: NDArray[np.float64]
    base: LocalShiftFunction | None = None

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

    def __call__(self, projected: NDArray[np.float64]) -> NDArray[np.float64]:
        """Local shifts of the projected (N, T, 2) points in nm."""
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
        shifts = (
            grid[tilt, :, iy0, ix0] * (1 - fy) * (1 - fx)
            + grid[tilt, :, iy0, ix1] * (1 - fy) * fx
            + grid[tilt, :, iy1, ix0] * fy * (1 - fx)
            + grid[tilt, :, iy1, ix1] * fy * fx
        )
        if self.base is not None:
            shifts = shifts + self.base(projected)
        return shifts


_RELION_ALIGNMENT_COLUMNS = (
    "rlnTomoXTilt",
    "rlnTomoYTilt",
    "rlnTomoZRot",
    "rlnTomoXShiftAngst",
    "rlnTomoYShiftAngst",
)


def _normalize_image_path(
    image_path: str | Path | Sequence[str | Path], ntilts: int
) -> Path | tuple[Path, ...]:
    if isinstance(image_path, (str, Path)):
        return Path(image_path)
    paths = tuple(Path(p) for p in image_path)
    if len(paths) != ntilts:
        raise ValueError(
            f"'image_path' must be a path or {ntilts} paths of each tilt, got "
            f"{len(paths)} paths."
        )
    return paths


def _same_relion_name(value: Any, name: str) -> bool:
    """Whether the rlnTomoName value (possibly parsed as a number) is the name."""
    if str(value) == name:
        return True
    try:
        return float(value) == float(name)
    except (TypeError, ValueError):
        return False


def _select_relion_tilt_series(df, tomo_name: str | None, path: Path) -> dict:
    """Select the row of the tilt series from the global data block."""
    names = df["rlnTomoName"].to_list()
    if tomo_name is None:
        if len(names) != 1:
            raise ValueError(
                f"{path} has {len(names)} tilt series. Specify one by 'tomo_name' from "
                f"{[str(n) for n in names]}."
            )
        return df.row(0, named=True)
    for i, name in enumerate(names):
        if _same_relion_name(name, tomo_name):
            return df.row(i, named=True)
    raise ValueError(
        f"Tilt series {tomo_name!r} is not found in {path}. Tilt series are "
        f"{[str(n) for n in names]}."
    )


def _select_relion_tilt_block(star, tomo_name: str | None, path: Path):
    """Select the data block of a tilt series in its STAR file."""
    if tomo_name is not None:
        for name, block in star.items():
            if _same_relion_name(name, tomo_name):
                return block
    for block in star.values():
        if "rlnMicrographName" in block.columns:
            return block
    raise ValueError(f"No tilt images (rlnMicrographName) are found in {path}.")


def _relion_project_dir(
    star_path: Path, relative: Path, project_dir: str | Path | None
) -> Path:
    """The RELION project directory, to which the paths in the STAR files are relative.

    If not given, the nearest parent directory of the STAR file where ``relative``
    exists is used.
    """
    if project_dir is not None:
        return Path(project_dir)
    if relative.is_absolute():
        return star_path.parent
    for parent in star_path.resolve().parents:
        if (parent / relative).exists():
            return parent
    raise ValueError(
        f"{relative} referred in {star_path} is not found in the parent directories. "
        "Give 'project_dir' explicitly."
    )


def _readonly(arr: NDArray) -> NDArray:
    arr = np.array(arr)  # copy
    arr.setflags(write=False)
    return arr


def _per_tilt(value, ntilts: int, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(ntilts, float(arr))
    if arr.shape != (ntilts,):
        raise ValueError(
            f"'{name}' must be a scalar or ({ntilts},) array, got {arr.shape}."
        )
    return _readonly(arr)


def _homogenize(points: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.concatenate([points, np.ones(points.shape[:-1] + (1,))], axis=-1)


def _rotation(angles: NDArray[np.float64], axis: str) -> NDArray[np.float64]:
    """(..., 4, 4) rotation matrices of (z, y, x, w) coordinates in degrees."""
    rad = np.deg2rad(angles)
    c, s = np.cos(rad), np.sin(rad)
    out = np.tile(np.eye(4), rad.shape + (1, 1))
    i, j = {"x": (0, 1), "y": (2, 0), "z": (1, 2)}[axis]
    out[..., i, i] = c
    out[..., i, j] = s
    out[..., j, i] = -s
    out[..., j, j] = c
    return out


def _translation(shifts: NDArray[np.float64]) -> NDArray[np.float64]:
    """(..., 4, 4) translation matrices of (z, y, x, w) coordinates."""
    out = np.tile(np.eye(4), shifts.shape[:-1] + (1, 1))
    out[..., :3, 3] = shifts
    return out


def _affine_to_nm(mtx: NDArray[np.float64]) -> NDArray[np.float64]:
    mtx = mtx.copy()
    mtx[:3, 3] /= _ANGSTROM_PER_NM
    return mtx


def _affine_to_angstrom(mtx: NDArray[np.float64]) -> NDArray[np.float64]:
    mtx = mtx.copy()
    mtx[:3, 3] *= _ANGSTROM_PER_NM
    return mtx


class _TorchShiftFunction:
    """A local shift function of torch (in Å) wrapped for numpy (in nm)."""

    def __init__(self, func):
        self.torch_function = func

    def __call__(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        import torch

        with torch.no_grad():
            tensor = torch.as_tensor(points * _ANGSTROM_PER_NM, dtype=torch.float32)
            out = self.torch_function(tensor)
        return out.cpu().numpy().astype(np.float64) / _ANGSTROM_PER_NM


class _NumpyShiftFunction:
    """A local shift function of numpy (in nm) wrapped for torch (in Å)."""

    def __init__(self, func: LocalShiftFunction):
        self.numpy_function = func

    def __call__(self, points):
        import torch
        import torch.nn.functional as F

        # projected points may be given in (z, y, x, w) for local_shifts_2d
        zyxw = points.ndim == 3 and points.shape[-1] == 4
        values = points[..., 1:3] if zyxw else points
        array = values.detach().cpu().numpy().astype(np.float64) / _ANGSTROM_PER_NM
        out = self.numpy_function(array) * _ANGSTROM_PER_NM
        out = torch.as_tensor(out, dtype=points.dtype, device=points.device)
        return F.pad(out, (1, 1)) if zyxw else out


def _shift_function_from_torch(func) -> LocalShiftFunction | None:
    from acryo.loader._tilt_refine import LocalShifts2D as TorchLocalShifts2D

    if func is None:
        return None
    if isinstance(func, _NumpyShiftFunction):
        return func.numpy_function
    if isinstance(func, TorchLocalShifts2D):
        return LocalShifts2D(
            func.control_points.detach().cpu().numpy() / _ANGSTROM_PER_NM,
            func.half_extent.detach().cpu().numpy() / _ANGSTROM_PER_NM,
            base=_shift_function_from_torch(func.base),
        )
    return _TorchShiftFunction(func)


def _shift_function_to_torch(func: LocalShiftFunction | None):
    import torch
    from acryo.loader._tilt_refine import LocalShifts2D as TorchLocalShifts2D

    if func is None:
        return None
    if isinstance(func, _TorchShiftFunction):
        return func.torch_function
    if isinstance(func, LocalShifts2D):
        return TorchLocalShifts2D(
            torch.as_tensor(func.control_points * _ANGSTROM_PER_NM).float(),
            torch.as_tensor(func.half_extent * _ANGSTROM_PER_NM).float(),
            base=_shift_function_to_torch(func.base),
        )
    return _NumpyShiftFunction(func)
