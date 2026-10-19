from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
import json
from typing import TYPE_CHECKING, Any, Sequence
import warnings
import polars as pl

import numpy as np
from numpy.typing import ArrayLike, NDArray

from acryo._types import nm
from acryo.ctf import PatchCTFCorrection, TiltSeriesCTF, _ImodDefocus, _warn_ignored
from acryo.projection._local_shifts import (
    AffineShifts2D,
    LocalAlignment,
    LocalShiftFunction,
    LocalShiftModel,
    _readonly,
    as_local_shift_model,
)

if TYPE_CHECKING:
    from typing_extensions import Self

_ANGSTROM_PER_NM = 10.0


@dataclass(frozen=True, eq=False)
class TiltSeries:
    """Alignment geometry of a tilt series, and the projection of 3D points.

    The geometry and the reconstruction from tilt series in :mod:`acryo.projection` are
    ported from `torch-reconstruct-tomogram
    <https://github.com/teamtomo/torch-reconstruct-tomogram>`_ and its tilt series
    model, `torch-tilt-series <https://github.com/teamtomo/torch-tilt-series>`_, in
    numpy and scipy. All the lengths are in nm instead of Å. Coordinates are in
    (z, y, x) order. Points in the tomogram space are projected to the detector by

    1. ``tomo2sample`` (tomogram space -> sample space),
    2. ``local_shifts`` (deformation of the sample, optional),
    3. ``sample2scope`` (rotation by ``x_tilts`` around the x axis, followed by the
       rotation by ``tilt_angles`` around the y axis),
    4. ``scope2detector`` (rotation by ``tilt_axis_angle`` around the optical axis,
       followed by the translation by ``sample_translations``),
    5. ``local_shifts_2d`` (shifts on the detector, optional), which may depend on
       the points in the sample space (see :class:`LocalShiftModel`).

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
    local_shifts_2d : LocalShiftModel or callable, optional
        Model of the shifts of the projected points on the detector, such as
        :class:`LocalShifts2D` and :class:`LocalAlignment`. A function that maps
        (N, T, 2) projected points (nm) to their shifts is converted to a
        :class:`FunctionShifts`, which cannot be serialized.
    image_path : str, Path or sequence of them, optional
        Path to the MRC file of the stack of the tilt images, or the paths to the MRC
        files of each tilt image (such as the motion-corrected tilt images of RELION).
    image_indices : (T,) array of int, optional
        Indices of the images in the stack corresponding to the tilts. If
        ``image_path`` is a sequence, indices of the images in each file (0 by
        default).
    scale : float, optional
        Pixel size of the tilt images in nm.
    ctf : TiltSeriesCTF, optional
        CTF parameters of each tilt. If given, the CTF of the tilt images is corrected
        by :class:`BackProjectionLoader` (see :meth:`ctf_correction`).
    """

    tilt_angles: NDArray[np.float64]
    tilt_axis_angle: NDArray[np.float64]
    sample_translations: NDArray[np.float64] = None  # type: ignore
    x_tilts: NDArray[np.float64] = 0.0  # type: ignore
    sample2levelled: NDArray[np.float64] = None  # type: ignore
    levelled2tomo: NDArray[np.float64] = None  # type: ignore
    local_shifts: LocalShiftFunction | None = None
    local_shifts_2d: LocalShiftModel | None = None
    image_path: Path | tuple[Path, ...] | None = None
    image_indices: NDArray[np.intp] | None = None
    scale: nm | None = None
    ctf: TiltSeriesCTF | None = None

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
        _set(self, "local_shifts_2d", as_local_shift_model(self.local_shifts_2d))
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
        if self.ctf is not None:
            if not isinstance(self.ctf, TiltSeriesCTF):
                raise TypeError(f"'ctf' must be a TiltSeriesCTF, got {type(self.ctf)}.")
            if self.ctf.ntilts != ntilts:
                raise ValueError(
                    f"'ctf' has the parameters of {self.ctf.ntilts} tilts, but the "
                    f"tilt series has {ntilts} tilts."
                )

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

    def select_tilts(self, indices: ArrayLike) -> Self:
        """Return the tilt series of the selected tilts.

        The alignment, ``local_shifts_2d``, ``image_path``, ``image_indices`` and
        ``ctf`` of the selected tilts are kept, so the selected tilts are projected and
        read from the same images as before.

        Parameters
        ----------
        indices : sequence of int
            Indices of the tilts to keep (such as ``tilt_angles[i]``), in the order of
            the new tilt series. They are not the indices of the images in the file.
        """
        idx = np.asarray(indices)
        if idx.ndim != 1 or idx.size == 0 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(
                f"'indices' must be a non-empty sequence of int, got {indices!r}."
            )
        idx = np.arange(self.ntilts)[idx]  # IndexError if out of range
        image_path = self.image_path
        image_indices = self.image_indices
        if image_indices is not None:
            image_indices = image_indices[idx]
        elif isinstance(image_path, Path):
            # all the images in the stack are the tilts
            image_indices = idx
        if isinstance(image_path, tuple):
            image_path = tuple(image_path[i] for i in idx)
        return self.replace(
            tilt_angles=self.tilt_angles[idx],
            tilt_axis_angle=self.tilt_axis_angle[idx],
            sample_translations=self.sample_translations[idx],
            x_tilts=self.x_tilts[idx],
            local_shifts_2d=(
                self.local_shifts_2d.select_tilts(idx)
                if self.local_shifts_2d is not None
                else None
            ),
            image_path=image_path,
            image_indices=image_indices,
            ctf=self.ctf.select_tilts(idx) if self.ctf is not None else None,
        )

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
        projected = _homogenize(points) @ matrices.reshape(-1, 4).T
        projected = projected.reshape(-1, self.ntilts, 2)
        if self.local_shifts_2d is not None:
            projected = projected + self.local_shifts_2d.shifts(self, points, projected)
        return projected.reshape(batch_shape + projected.shape[1:])

    def ctf_correction(self, hand: int = -1) -> PatchCTFCorrection:
        """The CTF correction of the tilt images by phase flipping of each patch.

        The returned filter can be added to :class:`TiltImagePreprocessor`. Each patch
        of the tilt images is phase-flipped by the CTF at the center of the patch. The
        defocus there is that of the plane z = 0 of the tomogram, which is the defocus
        of ``ctf`` at the projection of the tomogram center, changed by the depth of
        the plane in the microscope (by the tilt angle and the position). The local
        shifts are not considered.

        Parameters
        ----------
        hand : {1, -1}, default is -1
            Handedness of the defocus gradient (see
            :class:`acryo.ctf.PatchCTFCorrection`). -1 is the standard geometry, as
            ``rlnTomoHand = -1`` of RELION, or IMOD without ``InvertTiltAngles``.
        """
        if self.ctf is None:
            raise ValueError("'ctf' of the tilt series is not set.")
        mtx = self.projection_matrices @ self.tomo2sample  # (T, 4, 4)
        # The plane z = 0 of the tomogram is projected to the detector by `linear`
        # (relative to the projection of the tomogram center), and its depth is
        # `depth` @ (y, x).
        linear = mtx[:, 1:3, 1:3]
        depth = mtx[:, 0, 1:3]
        gradient = np.linalg.solve(linear.transpose(0, 2, 1), depth[..., np.newaxis])
        return PatchCTFCorrection(self.ctf, gradient[..., 0], mtx[:, 1:3, 3], hand=hand)

    @classmethod
    def from_etomo_directory(
        cls,
        path: str | Path,
        scale: nm,
        *,
        local_alignment: bool | str | Path = False,
        ctf: bool | str | Path = True,
    ) -> TiltSeries:
        """Read the alignment of an IMOD etomo directory.

        ``etomofiles`` is required. Excluded views are dropped. The tilt axis angle of
        each view is the rotation of the transformation (.xf) of the view, and the
        rest of the transformation, such as the change of magnification, is given as
        ``local_shifts_2d`` (:class:`AffineShifts2D`).

        Parameters
        ----------
        path : str or Path
            The etomo directory.
        scale : float
            Pixel size of the raw tilt images in nm.
        local_alignment : bool, str or Path, default is False
            Whether to read the local alignments of tiltalign as ``local_shifts_2d``
            (:class:`LocalAlignment`). If True, the file is ``LOCALFILE`` of tilt.com
            (or ``OutputLocalFile`` of align.com). A path to the file can also be
            given. As ``tilt`` of IMOD, the Z factors of the local alignments are used
            only if tilt.com has ``ZFACTORFILE``.
        ctf : bool, str or Path, default is True
            Whether to read the CTF parameters as ``ctf``, so that the CTF is
            corrected by :class:`BackProjectionLoader`. The defocus of each view is
            read from the defocus file of ctfplotter (``DefocusFile`` of
            ctfcorrection.com or ctfplotter.com, or ``<dataset>.defocus``), and the
            voltage, spherical aberration and amplitude contrast from
            ctfcorrection.com or ctfplotter.com. A path to the defocus file can also
            be given. If True and they are not found, a warning is issued and
            ``ctf`` is not set. The handedness of the defocus gradient is not read
            (see :meth:`ctf_correction`), and a warning is issued if
            ``InvertTiltAngles`` is set.
        """
        import etomofiles

        path = Path(path)
        df_all = etomofiles.read(path)
        df = df_all.loc[~df_all["excluded"]].reset_index(drop=True)
        image_path = Path(df["image_path"][0].split("[")[0])
        if not image_path.is_absolute():
            image_path = path / image_path
        xf = etomofiles.df_to_xf(df, yx=True)  # (T, 2, 3) in (y, x)
        # IMOD transforms the images to the aligned ones, which is the inverse of
        # the projection model.
        inv = np.linalg.inv(xf[:, :, :2])
        translations = -np.einsum("nij,nj->ni", inv, xf[:, :, 2]) * scale
        # The closest rotation of each view is the tilt axis angle, and the rest
        # (such as the change of magnification) is an affine transformation about the
        # projected center of the tomogram.
        u, _, vt = np.linalg.svd(inv)
        rotations = u @ vt
        tilt_axis_angle = np.rad2deg(np.arctan2(rotations[:, 0, 1], rotations[:, 0, 0]))
        affine = inv @ rotations.transpose(0, 2, 1)
        offsets = -np.einsum("nij,nj->ni", affine - np.eye(2), translations)
        # The sign of IMOD's XAXISTILT is opposite.
        xaxistilt = -(df["xaxistilt"].iloc[0] or 0.0)
        x_tilts = xaxistilt + np.nan_to_num(df["xtilt"].to_numpy(), nan=0.0)
        tilt_com = _read_imod_com(path / "tilt.com")
        if "ZFACTORFILE" in tilt_com:
            warnings.warn(
                "Z factors of the global alignment (ZFACTORFILE in tilt.com) are not "
                "supported and ignored.",
                UserWarning,
                stacklevel=2,
            )
        local_shifts_2d = None
        if local_alignment is not False:
            local_shifts_2d = _read_imod_local_alignment(
                path,
                local_alignment,
                tilt_com,
                image_path=image_path,
                scale=scale,
                indices=df["idx_tilt"].to_numpy(),
                nviews=len(df_all),
            )
        # the local alignments are increments to the aligned images, which are then
        # transformed by the inverse of .xf
        if not np.allclose(affine, np.eye(2), rtol=0, atol=1e-7):
            local_shifts_2d = AffineShifts2D(affine, offsets, base=local_shifts_2d)
        tilt_ctf = None
        if ctf is not False:
            tilt_ctf = _read_imod_ctf(
                path,
                ctf,
                image_path=image_path,
                indices=df["idx_tilt"].to_numpy(),
                tilt_axis_angle=tilt_axis_angle,
            )
        return cls(
            tilt_angles=df["tlt"].to_numpy(),
            tilt_axis_angle=tilt_axis_angle,
            sample_translations=translations,
            x_tilts=x_tilts,
            local_shifts_2d=local_shifts_2d,
            image_path=image_path,
            image_indices=df["idx_tilt"].to_numpy(),
            scale=scale,
            ctf=tilt_ctf,
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
        ctf: bool = True,
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
        ctf : bool, default is True
            Whether to read the CTF parameters as ``ctf``, so that the CTF is
            corrected by :class:`BackProjectionLoader`. The defocus of each tilt
            (``rlnDefocusU``, ``rlnDefocusV`` and ``rlnDefocusAngle``) is read from
            the tilt series, and ``rlnVoltage``, ``rlnSphericalAberration`` and
            ``rlnAmplitudeContrast`` from the global data block. If they are not
            found, a warning is issued and ``ctf`` is not set. The handedness of the
            defocus gradient is not read (see :meth:`ctf_correction`), and a warning
            is issued if ``rlnTomoHand`` is 1.
        """
        from starfile_rs import read_star

        path = Path(path)
        star = read_star(path)
        series: dict[str, Any] | None = None
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
            ctf=_read_relion_ctf(df, series, path) if ctf else None,
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
        local_shifts_2d = data.get("local_shifts_2d")
        ctf = data.get("ctf")
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
            local_shifts_2d=(
                LocalShiftModel.from_dict(local_shifts_2d)
                if local_shifts_2d is not None
                else None
            ),
            image_path=image_path,
            image_indices=image_indices,
            scale=data["scale"],
            ctf=TiltSeriesCTF.from_dict(ctf) if ctf is not None else None,
        )

    def dump_json(self) -> str:
        if self.local_shifts is not None:
            raise TypeError(
                f"'local_shifts' {self.local_shifts!r} cannot be serialized."
            )
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
                "local_shifts_2d": (
                    self.local_shifts_2d.to_dict()
                    if self.local_shifts_2d is not None
                    else None
                ),
                "image_path": image_path,
                "image_indices": image_indices,
                "scale": self.scale,
                "ctf": self.ctf.to_dict() if self.ctf is not None else None,
            }
        )


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


def _read_imod_com(path: Path) -> dict[str, list[str]]:
    """Read the options (in upper case) and their values of an IMOD command file."""
    if not path.exists():
        return {}
    options: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts or parts[0].startswith(("#", "$")):
            continue
        options[parts[0].upper()] = parts[1:]
    return options


def _read_imod_local_alignment(
    path: Path,
    local_alignment: bool | str | Path,
    tilt_com: dict[str, list[str]],
    *,
    image_path: Path,
    scale: nm,
    indices: NDArray[np.integer],
    nviews: int,
) -> LocalAlignment | None:
    """Read the local alignments of an etomo directory for the selected views."""
    if local_alignment is True:
        names = tilt_com.get("LOCALFILE") or _read_imod_com(path / "align.com").get(
            "OUTPUTLOCALFILE"
        )
        if names and Path(names[0]).exists():
            file = Path(names[0])
        else:
            return None
    else:
        file = Path(local_alignment)
    if not file.is_absolute():
        file = path / file
    if fullimage := tilt_com.get("FULLIMAGE"):
        image_shape = (int(fullimage[1]), int(fullimage[0]))
    else:
        from acryo.projection._io import MrcImageStack

        image_shape = MrcImageStack(image_path).shape[1:]
    local = LocalAlignment.from_imod(
        file, image_shape, scale, z_factors="ZFACTORFILE" in tilt_com
    )
    if local.ntilts == nviews:
        return local.select_tilts(indices)
    elif local.ntilts == len(indices):
        return local
    raise ValueError(
        f"{file} has the local alignments of {local.ntilts} views, but the tilt series "
        f"has {nviews} views ({len(indices)} views are not excluded)."
    )


_IMOD_CTF_OPTIONS = ("VOLTAGE", "SPHERICALABERRATION", "AMPLITUDECONTRAST")


def _read_imod_ctf(
    path: Path,
    ctf: bool | str | Path,
    *,
    image_path: Path,
    indices: NDArray[np.integer],
    tilt_axis_angle: NDArray[np.float64],
) -> TiltSeriesCTF | None:
    """Read the CTF parameters of the selected views from an etomo directory.

    None is returned with a warning if ``ctf`` is True and they are not found.
    """
    explicit = ctf is not True
    plotter_com = _read_imod_com(path / "ctfplotter.com")
    # ctfcorrection.com (ctfphaseflip) is the one used for the correction
    params = {**plotter_com, **_read_imod_com(path / "ctfcorrection.com")}
    if explicit:
        file = Path(ctf)
    elif names := params.get("DEFOCUSFILE"):
        file = Path(names[0])
    else:
        file = Path(f"{image_path.stem}.defocus")
    if not file.is_absolute():
        file = path / file
    missing = [key for key in _IMOD_CTF_OPTIONS if not params.get(key)]
    if not file.exists() or missing:
        if not file.exists():
            msg = f"Defocus file {file} is not found"
        else:
            msg = (
                f"{', '.join(missing)} not found in ctfcorrection.com or "
                f"ctfplotter.com in {path}"
            )
        if explicit:
            raise (ValueError if file.exists() else FileNotFoundError)(f"{msg}.")
        warnings.warn(
            f"{msg}, so the CTF parameters are not read. Run ctfplotter in etomo, "
            "or give 'ctf=False' to suppress this warning.",
            UserWarning,
            stacklevel=3,
        )
        return None
    defocus = _ImodDefocus.read(file)
    if np.any(defocus.phase_shift != 0):
        _warn_ignored(f"Phase shifts in {file} are not supported and ignored.")
    defocus1, defocus2, angle = defocus.at_views(np.asarray(indices) + 1)
    stack = plotter_com.get("INPUTSTACK")
    if stack and stack[0].endswith(".ali"):
        # astigmatism angles of the aligned stack, whose tilt axis is vertical
        angle = angle + tilt_axis_angle
    invert = params.get("INVERTTILTANGLES")
    if invert is not None and (not invert or invert[0] != "0"):
        _warn_ignored(_inverted_hand_message(f"InvertTiltAngles in {path}"))
    return TiltSeriesCTF(
        defocus_u=-defocus1 / 1000,  # nm (underfocus positive) -> μm
        defocus_v=-defocus2 / 1000,
        astigmatism_angle=angle,
        voltage=float(params["VOLTAGE"][0]),
        spherical_aberration=float(params["SPHERICALABERRATION"][0]),
        amplitude_contrast=float(params["AMPLITUDECONTRAST"][0]),
    )


_RELION_DEFOCUS_COLUMNS = ("rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle")
_RELION_OPTICS_COLUMNS = (
    "rlnVoltage",
    "rlnSphericalAberration",
    "rlnAmplitudeContrast",
)


def _read_relion_ctf(
    df: pl.DataFrame,
    series: dict[str, Any] | None,
    path: Path,
) -> TiltSeriesCTF | None:
    """Read the CTF parameters from the tilt series and its row in the global block.

    None is returned with a warning if they are not found.
    """
    if not all(c in df.columns for c in _RELION_DEFOCUS_COLUMNS):
        msg = f"{', '.join(_RELION_DEFOCUS_COLUMNS)} are not found in {path}"
    elif series is None or any(series.get(c) is None for c in _RELION_OPTICS_COLUMNS):
        msg = (
            f"{', '.join(_RELION_OPTICS_COLUMNS)} are not found in the global data "
            f"block of {path}"
        )
    else:
        msg = None
    if msg is not None:
        warnings.warn(
            f"{msg}, so the CTF parameters are not read. Read the STAR file of the "
            "tilt series after CTF estimation (such as tilt_series.star of a "
            "CtfFind job), or give 'ctf=False' to suppress this warning.",
            UserWarning,
            stacklevel=3,
        )
        return None
    assert series is not None
    if "rlnPhaseShift" in df.columns and np.any(df["rlnPhaseShift"].to_numpy() != 0):
        _warn_ignored(f"Phase shifts in {path} are not supported and ignored.")
    if (hand := series.get("rlnTomoHand")) is not None and float(hand) > 0:
        _warn_ignored(_inverted_hand_message(f"rlnTomoHand = {hand} in {path}"))
    angst2um = 1e-4
    return TiltSeriesCTF(
        defocus_u=-df["rlnDefocusU"].to_numpy().astype(np.float64) * angst2um,
        defocus_v=-df["rlnDefocusV"].to_numpy().astype(np.float64) * angst2um,
        astigmatism_angle=df["rlnDefocusAngle"].to_numpy().astype(np.float64),
        voltage=float(series["rlnVoltage"]),
        spherical_aberration=float(series["rlnSphericalAberration"]),
        amplitude_contrast=float(series["rlnAmplitudeContrast"]),
    )


def _inverted_hand_message(source: str) -> str:
    return (
        f"The defocus gradient is inverted by {source}, but the handedness is not "
        "read. Give the CTF correction with hand=1 to BackProjectionLoader, such as "
        "preprocess=TiltImagePreprocessor().with_filters(tilt_series.ctf_correction("
        "hand=1))."
    )


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
