# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any, Iterable, SupportsIndex
import warnings

import numpy as np
from numpy.typing import NDArray
from dask import array as da

from acryo._types import nm, pixel
from acryo._dask import DaskArrayList, DaskTaskPool
from acryo.backend import Backend
from acryo.loader import _misc
from acryo.loader._base import LoaderBase, Unset, _ShapeType
from acryo.molecules import Molecules
from acryo.tilt import TiltSeriesModel, single_axis

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self
    from torch_tilt_series import TiltSeries

_ANGSTROM_PER_NM = 10.0


class PseudoSubtomogramLoader(LoaderBase):
    """A loader that reconstructs subtomograms directly from a tilt series.

    No tomogram is needed. Every time subtomograms are requested, they are
    reconstructed from the tilt series at the pixel size given by ``scale`` using
    `torch-reconstruct-tomogram <https://github.com/teamtomo/torch-reconstruct-tomogram>`_
    (install by ``pip install acryo[torch]``). CTF is not corrected.

    When subtomograms are requested, cubes aligned to the tomogram axes are
    reconstructed immediately using the multi-threaded PyTorch. A cube is
    reconstructed for each molecule, or, if the molecules are dense, for each chunk of
    about ``chunk_size`` that contains molecules, whichever has smaller total volume.
    The subtomograms are then cropped from the cubes and rotated to the orientation of
    the molecules in parallel by dask.

    Parameters
    ----------
    tilt_series : torch_tilt_series.TiltSeries
        Tilt series with the alignment parameters. ``image_path`` and
        ``pixel_spacing`` must be set. The tilt images are loaded (and preprocessed)
        only once when they are needed for the first time.
    molecules : Molecules
        Molecules object that represents positions and orientations of
        subtomograms.
    order : int, default is 3
        Interpolation order used to rotate the reconstructed subtomograms.
    scale : float, default is 1.0
        Pixel size of the reconstructed subtomograms in nm.
    output_shape : int or tuple of int, optional
        Shape of output subtomogram in pixel. This parameter is not required
        if template (or mask) image is available immediately.
    corner_safe : bool, default is False
        If true, regions larger than ``output_shape`` are reconstructed so that
        densities at the corners will not be lost due to rotation.
    tomogram_center : (float, float, float), default is (0, 0, 0)
        The (z, y, x) coordinates of the tomogram center in nm, in the same
        coordinate system as the molecules. The tomogram center is the origin of the
        tomogram space of the tilt series. For example, if the molecules were picked
        in a tomogram of shape ``(D, H, W)`` with pixel size ``s`` nm reconstructed
        by ``torch_reconstruct_tomogram.reconstruct_tomogram``, this is
        ``(D // 2 * s, H // 2 * s, W // 2 * s)``.
    tilt_model : TiltSeriesModel, optional
        Tilt series model used for alignment. By default, a single-axis model around
        the y-axis is created from the tilt angles of the tilt series (and
        ``tomo_hand``).
    preprocess : bool, default is True
        If true, tilt images are preprocessed by
        ``torch_tilt_series.preprocess_tilt_series_images`` before reconstruction.
    preprocess_kwargs : dict, optional
        Keyword arguments passed to ``preprocess_tilt_series_images``.
    batch_size : int, optional
        Maximum number of subtomograms reconstructed in one call of PyTorch. Memory
        usage is proportional to this value. If None, all the subtomograms are
        reconstructed in one call. Chunks are always reconstructed one by one.
    chunk_size : float, default is 100.0
        Approximate size of the chunks in nm, including the overlap of the
        subtomogram size.
    tomo_hand : int, default is 1
        Handedness of the tomogram, 1 or -1. The handedness is inverted depending on
        the positive direction of the tilt angles. If -1, the tomogram of the
        molecules is regarded as the one reconstructed from the tilt series, mirrored
        along the z axis around ``tomogram_center``.
    """

    def __init__(
        self,
        tilt_series: TiltSeries,
        molecules: Molecules,
        order: int = 3,
        scale: nm = 1.0,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
        tomogram_center: tuple[nm, nm, nm] = (0.0, 0.0, 0.0),
        tilt_model: TiltSeriesModel | None = None,
        preprocess: bool = True,
        preprocess_kwargs: dict[str, Any] | None = None,
        batch_size: int | None = None,
        chunk_size: nm = 100.0,
        tomo_hand: int = 1,
    ) -> None:
        TiltSeries = _import_tilt_series_class()
        if not isinstance(tilt_series, TiltSeries):
            raise TypeError(
                "The first argument 'tilt_series' must be a TiltSeries object, got "
                f"{type(tilt_series)}."
            )
        if tilt_series.image_path is None:
            raise ValueError("'image_path' of the tilt series is not set.")
        tilt_series.pixel_spacing  # raises if not set
        if not isinstance(molecules, Molecules):
            raise TypeError(
                "The second argument 'molecules' must be a Molecules object, got"
                f"{type(molecules)}."
            )
        _center = np.asarray(tomogram_center, dtype=np.float64)
        if _center.shape != (3,):
            raise ValueError(
                f"'tomogram_center' must be a (z, y, x) tuple, got {tomogram_center!r}."
            )
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"'batch_size' must be positive, got {batch_size}.")
        if chunk_size <= 0:
            raise ValueError(f"'chunk_size' must be positive, got {chunk_size}.")
        if tomo_hand not in (1, -1):
            raise ValueError(f"'tomo_hand' must be 1 or -1, got {tomo_hand!r}.")
        if tilt_model is None:
            # mirroring along the z axis inverts the tilt angles
            angles = tilt_series.tilt_angles * tomo_hand
            tilt_model = single_axis(
                (float(angles.min()), float(angles.max())), axis="y"
            )
        self._images = _TiltSeriesImages(tilt_series, preprocess, preprocess_kwargs)
        self._molecules = molecules
        self._corner_safe = corner_safe
        self._tomogram_center = _center
        self._tilt_model = tilt_model
        self._batch_size = None if batch_size is None else int(batch_size)
        self._chunk_size = float(chunk_size)
        self._tomo_hand = int(tomo_hand)
        super().__init__(order=order, scale=scale, output_shape=output_shape)

    def __repr__(self) -> str:
        ntilts = self.tilt_series.tilt_angles.shape[0]
        mole_repr = repr(self.molecules)
        return (
            f"{self.__class__.__name__}(tilt_series=<{ntilts} tilts>, "
            f"molecules={mole_repr}, output_shape={self.output_shape}, "
            f"order={self.order}, scale={self.scale:.4f})"
        )

    @classmethod
    def from_etomo_directory(
        cls,
        path,
        molecules: Molecules,
        order: int = 3,
        scale: nm = 1.0,
        corner_safe: bool = False,
        device: torch.device | str = "cpu",
        tomogram_center: tuple[nm, nm, nm] = (0.0, 0.0, 0.0),
        preprocess: bool = True,
        preprocess_kwargs: dict[str, Any] | None = None,
        batch_size: int | None = None,
        chunk_size: nm = 100.0,
        tomo_hand: int = 1,
    ) -> PseudoSubtomogramLoader:
        from torch_tilt_series.io import from_etomo_directory

        ts = from_etomo_directory(
            path,
            pixel_spacing=scale * _ANGSTROM_PER_NM,
            device=device,
        )
        return PseudoSubtomogramLoader(
            ts,
            molecules=molecules,
            order=order,
            scale=scale,
            corner_safe=corner_safe,
            tomogram_center=tomogram_center,
            preprocess=preprocess,
            preprocess_kwargs=preprocess_kwargs,
            batch_size=batch_size,
            chunk_size=chunk_size,
            tomo_hand=tomo_hand,
        )

    @property
    def tilt_series(self) -> TiltSeries:
        """Return the tilt series."""
        return self._images.tilt_series

    @property
    def molecules(self) -> Molecules:
        """Return the molecules of the subtomogram loader."""
        return self._molecules

    @property
    def corner_safe(self) -> bool:
        """Return whether the loader is corner-safe."""
        return self._corner_safe

    @property
    def tomogram_center(self) -> NDArray[np.float64]:
        """Return the (z, y, x) coordinates of the tomogram center in nm."""
        return self._tomogram_center

    @property
    def tilt_model(self) -> TiltSeriesModel:
        """Return the tilt model of the subtomogram loader."""
        return self._tilt_model

    @property
    def batch_size(self) -> int | None:
        """Return the maximum number of subtomograms reconstructed at once."""
        return self._batch_size

    @property
    def chunk_size(self) -> nm:
        """Return the approximate size of the chunks in nm."""
        return self._chunk_size

    @property
    def tomo_hand(self) -> int:
        """Return the handedness of the tomogram (1 or -1)."""
        return self._tomo_hand

    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset | None = None,
        order: int | None = None,
        scale: float | None = None,
    ) -> Self:
        """Return a new instance with different parameter(s)."""
        if molecules is None:
            molecules = self.molecules
        if output_shape is None:
            output_shape = self.output_shape
        if order is None:
            order = self.order
        if scale is None:
            scale = self.scale
        out = self.__class__(
            self.tilt_series,
            molecules=molecules,
            order=order,
            scale=scale,
            output_shape=output_shape,
            corner_safe=self.corner_safe,
            tomogram_center=tuple(self.tomogram_center),
            tilt_model=self.tilt_model,
            batch_size=self.batch_size,
            chunk_size=self.chunk_size,
            tomo_hand=self.tomo_hand,
        )
        out._images = self._images  # share the loaded tilt images
        return out

    def binning(self, binsize: float = 2, *, compute: bool = True) -> Self:
        """Return a new instance with larger pixel size.

        Subtomograms are reconstructed at the binned pixel size, so ``compute`` has
        no effect. It is only for the compatibility with ``SubtomogramLoader``.
        """
        if binsize == 1:
            return self.copy()
        return self.replace(scale=self.scale * binsize)

    def construct_loading_tasks(
        self,
        output_shape: _ShapeType = None,
        backend: Backend | None = None,
    ) -> DaskArrayList:
        """Construct a list of subtomogram lazy loader.

        The subtomograms are reconstructed immediately, and only their rotation is
        delayed.

        Returns
        -------
        DaskArrayList
            Each dask array returns a subtomogram on computation.
        """
        output_shape = self._get_output_shape(output_shape)
        if self.molecules.count() == 0:
            return DaskArrayList([])
        xp = backend or Backend()
        cubes, cube_index, regions, matrices = self._reconstruct(output_shape)
        pool = DaskTaskPool.from_func(_rotated_crop)
        for i, region in enumerate(regions):
            subimg = cubes[cube_index[i]][region]
            pool.add_task(subimg, matrices[i], output_shape, self.order, xp)
        return pool.asarrays(shape=output_shape, dtype=np.float32)

    def construct_dask(
        self,
        output_shape: pixel | tuple[pixel, ...] | None = None,
        backend: Backend | None = None,
    ) -> da.Array:
        """Construct a dask array of subtomograms.

        The subtomograms are reconstructed immediately, and only their rotation is
        delayed.

        Returns
        -------
        da.Array
            An 4-D array which ``arr[i]`` corresponds to the ``i``-th subtomogram.
        """
        output_shape = self._get_output_shape(output_shape)
        if self.molecules.count() == 0:
            return da.zeros((0,) + tuple(output_shape), dtype=np.float32)
        return super().construct_dask(output_shape, backend)

    def load(
        self,
        idx: SupportsIndex | slice | Iterable[SupportsIndex],
        output_shape: _ShapeType = None,
    ) -> NDArray[np.float32]:
        """Load subtomogram(s) of given index.

        Only the subtomograms of given index are reconstructed.

        Parameters
        ----------
        idx : int, slice or iterable of int
            Subtomogram index.
        output_shape : int or tuple of int, optional
            Shape of the output subtomograms.

        Returns
        -------
        3D array or 4D array
            Subtomogram(s) of given index.
        """
        if isinstance(idx, SupportsIndex):
            return self.load([idx], output_shape=output_shape)[0]
        elif isinstance(idx, slice):
            spec = idx
        elif hasattr(idx, "__iter__"):
            spec = [i.__index__() for i in idx]
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")
        loader = self.replace(molecules=self.molecules.subset(spec))
        return loader.asnumpy(output_shape=output_shape)

    def _reconstruct(
        self,
        output_shape: tuple[pixel, pixel, pixel],
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.intp],
        list[tuple[slice, ...]],
        NDArray[np.float32],
    ]:
        """Reconstruct cubes that contain the subtomograms.

        A cube is reconstructed for each molecule, or, if the molecules are dense,
        for each chunk of about ``chunk_size`` that contains molecules. The strategy
        with smaller total volume is chosen.

        Returns
        -------
        cubes : (M, L, L, L) array
            The reconstructed cubes.
        cube_index : (N,) array
            Index of the cube that contains each molecule.
        regions : list of tuple of slices
            Region of the cube that each subtomogram is cropped from.
        matrices : (N, 4, 4) array
            Affine matrices that map the output coordinates to the coordinates of the
            cropped regions.
        """
        if self.corner_safe:
            extent = math.sqrt(sum(s**2 for s in output_shape))
        else:
            extent = max(output_shape)
        output_spacing = self.scale * _ANGSTROM_PER_NM
        input_spacing = self.tilt_series.pixel_spacing
        mole = self.molecules
        nmole = mole.count()
        pos = mole.pos.astype(np.float64)
        rotations = mole.rotator.as_matrix()
        if self.tomo_hand == -1:
            # The tomogram of the molecules is the true one mirrored along the z axis
            # around the tomogram center. Convert the molecules to the true tomogram.
            pos[:, 0] = 2 * self.tomogram_center[0] - pos[:, 0]
            rotations[:, 0, :] *= -1

        # strategy 1: reconstruct a cube for each molecule
        sidelength, pixel_spacing = _cube_geometry(
            extent, 2 * self.order, output_spacing, input_spacing
        )
        centers = pos
        cube_index = np.arange(nmole)
        batch_size = self.batch_size or nmole

        # strategy 2: reconstruct a cube for each chunk that contains molecules. Each
        # chunk has margins so that it contains the whole subtomograms.
        margin = 2 * self.order + 2
        core = math.floor(self.chunk_size / self.scale) - math.ceil(extent) - margin
        if core > 0:
            chunk_sidelength, chunk_spacing = _cube_geometry(
                core + extent, margin, output_spacing, input_spacing
            )
            origin = pos.min(axis=0)
            core_nm = core * self.scale
            chunk_ids = np.floor((pos - origin) / core_nm).astype(np.intp)
            unique_ids, inverse = np.unique(chunk_ids, axis=0, return_inverse=True)
            volume_per_molecule = nmole * (sidelength * pixel_spacing) ** 3
            volume_per_chunk = (
                unique_ids.shape[0] * (chunk_sidelength * chunk_spacing) ** 3
            )
            if volume_per_chunk < volume_per_molecule:
                sidelength, pixel_spacing = chunk_sidelength, chunk_spacing
                centers = origin + (unique_ids + 0.5) * core_nm
                cube_index = inverse.ravel()
                batch_size = 1

        cubes = self._reconstruct_cubes(centers, sidelength, pixel_spacing, batch_size)

        # Crop the region around each molecule from the cube, similar to
        # `SubtomogramLoader`. The center of the cubes (sidelength // 2) is at
        # `centers`, and the pixel spacing of the cubes is slightly different from
        # the output.
        stretch = output_spacing / pixel_spacing
        mole_centers = sidelength // 2 + (pos - centers[cube_index]) * (
            _ANGSTROM_PER_NM / pixel_spacing
        )
        half = extent * stretch / 2 + self.order
        starts = np.clip(np.floor(mole_centers - half), 0, sidelength).astype(np.intp)
        stops = np.clip(np.ceil(mole_centers + half) + 1, 0, sidelength).astype(np.intp)
        regions = [
            tuple(slice(start, stop) for start, stop in zip(start_i, stop_i))
            for start_i, stop_i in zip(starts.tolist(), stops.tolist())
        ]
        translation_0 = np.tile(np.eye(4, dtype=np.float32), (nmole, 1, 1))
        translation_0[:, :3, 3] = mole_centers - starts
        rot_mat = np.tile(np.eye(4, dtype=np.float32), (nmole, 1, 1))
        rot_mat[:, :3, :3] = rotations * stretch
        translation_1 = np.eye(4, dtype=np.float32)
        translation_1[:3, 3] = -(np.array(output_shape) / 2 - 0.5)
        matrices = np.matmul(np.matmul(translation_0, rot_mat), translation_1)
        return cubes, cube_index, regions, matrices

    def _reconstruct_cubes(
        self,
        centers: NDArray[np.float64],
        sidelength: int,
        pixel_spacing: float,
        batch_size: int,
    ) -> NDArray[np.float32]:
        """Reconstruct cubes of given sidelength and pixel spacing (Å) at centers."""
        import torch
        from torch_reconstruct_tomogram.reconstruct import _reconstruct_subvolume

        points = ((centers - self.tomogram_center) * _ANGSTROM_PER_NM).astype(
            np.float32
        )
        # NOTE: The public `reconstruct_subvolume` loads the tilt images from the file
        # every time, so the private function is used here.
        images = self._images.get()
        ncubes = points.shape[0]
        cubes = np.empty((ncubes,) + (sidelength,) * 3, dtype=np.float32)
        with torch.no_grad():
            for start in range(0, ncubes, batch_size):
                sl = slice(start, start + batch_size)
                cubes[sl] = (
                    -_reconstruct_subvolume(
                        self.tilt_series,
                        images,
                        points[sl],
                        sidelength,
                        output_pixel_spacing=pixel_spacing,
                    )
                    .cpu()
                    .numpy()
                )
        return cubes

    def reconstruct_tomogram(
        self,
        shape: pixel | tuple[pixel, pixel, pixel],
        *,
        patch_size: pixel = 64,
        blend_margin: pixel | None = None,
        batch_size: int | None = 4,
    ) -> NDArray[np.float32]:
        """Reconstruct the tomogram from the tilt series.

        The tomogram is reconstructed at the pixel size of ``scale`` using
        ``torch_reconstruct_tomogram.reconstruct_tomogram``. The tomogram is centered
        at ``tomogram_center``, that is, the voxel ``(i, j, k)`` is at
        ``tomogram_center + ((i, j, k) - shape // 2) * scale`` nm in the coordinate
        system of the molecules. If ``tomogram_center`` is ``shape // 2 * scale``,
        ``SubtomogramLoader(tomogram, loader.molecules, scale=loader.scale)`` loads
        the same subtomograms as this loader. If ``tomo_hand`` is -1, the tomogram is
        mirrored along the z axis around the center.

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the tomogram in pixel.
        patch_size : int, default is 64
            Distance between the centers of the reconstructed patches in pixel. Must
            be even.
        blend_margin : int, optional
            Margin of each patch in pixel, which is blended with the neighboring
            patches. Default is ``patch_size // 4``. This value may be slightly
            changed to avoid odd-sized crops of the tilt images, which shift the
            patches by one pixel in ``torch-reconstruct-tomogram``.
        batch_size : int, default is 4
            Number of patches reconstructed at once. Larger value is faster but uses
            more memory. If None, all the patches are reconstructed at once.

        Returns
        -------
        np.ndarray
            The reconstructed tomogram.
        """
        from torch_reconstruct_tomogram import reconstruct_tomogram

        shape = _misc.normalize_shape(shape, ndim=3)
        if patch_size <= 0 or patch_size % 2 == 1:
            raise ValueError(
                f"'patch_size' must be positive and even, got {patch_size}."
            )
        if blend_margin is None:
            blend_margin = patch_size // 4
        if blend_margin < 0:
            raise ValueError(f"'blend_margin' must be >= 0, got {blend_margin}.")
        output_spacing = self.scale * _ANGSTROM_PER_NM
        blend_margin = _even_crop_margin(
            patch_size,
            blend_margin,
            output_spacing=output_spacing,
            input_spacing=self.tilt_series.pixel_spacing,
        )
        depth = shape[0]
        if self.tomo_hand == -1:
            # The center (depth // 2) must be the center of the mirrored tomogram, so
            # the depth is made odd before mirroring.
            shape = (2 * (depth // 2) + 1,) + shape[1:]
        tomogram = reconstruct_tomogram(
            self.tilt_series,
            shape,
            sidelength=patch_size,
            batch_size=batch_size,
            output_pixel_spacing=output_spacing,
            preprocess=self._images.preprocess,
            blend_margin=blend_margin,
            **self._images.preprocess_kwargs,
        )
        if self.tomo_hand == -1:
            tomogram = tomogram.flip(0)[:depth]
        return tomogram.cpu().numpy()

    def _default_align_kwargs(self) -> dict[str, Any]:
        """Return default keyword arguments for alignment."""
        return {"tilt": self.tilt_model}


class _TiltSeriesImages:
    """Tilt series and its images that are loaded only once on demand."""

    def __init__(
        self,
        tilt_series: TiltSeries,
        preprocess: bool,
        preprocess_kwargs: dict[str, Any] | None,
    ):
        self.tilt_series = tilt_series
        self.preprocess = preprocess
        self.preprocess_kwargs = dict(preprocess_kwargs or {})
        self._images: torch.Tensor | None = None
        self._lock = threading.Lock()

    def get(self) -> torch.Tensor:
        """Get the (preprocessed) tilt images."""
        if self._images is None:
            with self._lock:
                if self._images is None:
                    self._images = self._load()
        return self._images

    def _load(self) -> torch.Tensor:
        from torch_tilt_series import (
            load_tilt_series_images,
            preprocess_tilt_series_images,
        )

        images = load_tilt_series_images(self.tilt_series)
        if self.preprocess:
            images = preprocess_tilt_series_images(images, **self.preprocess_kwargs)
        return images


def _rotated_crop(
    cube: NDArray[np.float32],
    matrix: NDArray[np.float32],
    output_shape: tuple[int, int, int],
    order: int,
    backend: Backend,
) -> NDArray[np.float32]:
    """Rotate the reconstructed cube to the output subvolume."""
    return backend.rotated_crop(
        backend.asarray(cube),
        matrix,
        shape=output_shape,
        order=order,
        cval=backend.mean,
    )


_PAD_FACTOR = 2.0


def _cube_geometry(
    extent: float,
    margin: int,
    output_spacing: float,
    input_spacing: float,
) -> tuple[int, float]:
    """Determine the sidelength and the pixel spacing (Angstrom) of the cube.

    ``torch-reconstruct-tomogram`` crops the tilt images around the projected point
    at the native pixel spacing and Fourier-rescales the crops. The size of the
    crops is rounded to an integer, and odd-sized crops are shifted by one pixel.
    To avoid both, the pixel spacing of the cube is adjusted so that the crop size
    is exactly an even integer. The cube is large enough to sample a region of
    ``extent`` output pixels with ``margin`` pixels.
    """
    ratio = output_spacing / input_spacing
    sidelength = math.ceil(extent) + margin
    while True:
        sidelength += sidelength % 2  # odd sidelength is not supported
        padded = int(_PAD_FACTOR * sidelength)
        native = max(2 * round(padded * ratio / 2), 2)
        pixel_spacing = input_spacing * native / padded
        if math.ceil(extent * output_spacing / pixel_spacing) + margin <= sidelength:
            return sidelength, pixel_spacing
        sidelength += 1


def _even_crop_margin(
    patch_size: int,
    blend_margin: int,
    output_spacing: float,
    input_spacing: float,
) -> int:
    """Find the blend margin closest to the given one with even-sized crops.

    In ``torch_reconstruct_tomogram.reconstruct_tomogram``, the tilt images are
    cropped for each patch of size ``patch_size + 2 * blend_margin``. Odd-sized crops
    are shifted by one pixel, so the blend margin is changed to avoid them.
    """
    for diff in range(max(blend_margin, 8) + 1):
        for margin in (blend_margin + diff, blend_margin - diff):
            if margin < 0:
                continue
            padded = int(_PAD_FACTOR * (patch_size + 2 * margin))
            if round(padded * output_spacing / input_spacing) % 2 == 0:
                return margin
    warnings.warn(
        "Could not avoid odd-sized crops of the tilt images. The tomogram may be "
        "shifted by up to one pixel.",
        UserWarning,
        stacklevel=3,
    )
    return blend_margin


def _import_tilt_series_class() -> type[TiltSeries]:
    try:
        import torch_reconstruct_tomogram  # noqa: F401
        from torch_tilt_series import TiltSeries
    except ImportError as e:
        raise ImportError(
            "PseudoSubtomogramLoader requires torch-reconstruct-tomogram. Install it "
            "by `pip install acryo[torch]`."
        ) from e
    return TiltSeries
