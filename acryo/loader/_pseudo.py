# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import copy
import math
import threading
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence, SupportsIndex
import uuid
import warnings

import numpy as np
from numpy.typing import NDArray
from dask import array as da
from dask.highlevelgraph import HighLevelGraph

from acryo._types import nm, pixel
from acryo._dask import DaskArrayList
from acryo.backend import Backend
from acryo.loader import _misc
from acryo.loader._base import LoaderBase, Unset, _ShapeType, TemplateInputType
from acryo.molecules import Molecules
from acryo.projection import DirectFourierReconstructor
from acryo.tilt import TiltSeriesModel, single_axis

if TYPE_CHECKING:
    import torch
    from typing_extensions import Self
    from torch_tilt_series import TiltSeries

_ANGSTROM_PER_NM = 10.0


class PseudoSubtomogramLoader(LoaderBase):
    """A loader that reconstructs subtomograms directly from a tilt series.

    No tomogram is needed. Every time subtomograms are requested, they are
    reconstructed from the tilt series at the pixel size given by ``scale`` by direct
    Fourier inversion (insertion of the central slices into the 3D Fourier
    transform), as in ``torch-reconstruct-tomogram``. A cube aligned to the tomogram
    axes is reconstructed around each molecule, and the subtomogram is cropped from
    the cube in the same way as ``SubtomogramLoader``. Subtomograms are reconstructed
    lazily in dask tasks. CTF is not corrected. The tilt series is given as a
    ``TiltSeries`` of
    `torch-tilt-series <https://github.com/teamtomo/torch-tilt-series>`_ (install by
    ``pip install acryo[torch]``).

    Parameters
    ----------
    tilt_series : torch_tilt_series.TiltSeries
        Tilt series with the alignment parameters. ``image_path`` and
        ``pixel_spacing`` must be set. The tilt images are loaded only once when they
        are needed for the first time. They are read from the file tilt by tilt,
        preprocessed at the original pixel spacing, binned by the largest integer
        factor whose pixel spacing does not exceed ``scale``, and cached in memory.
        The raw tilt series is never in memory all at once.
    molecules : Molecules
        Molecules object that represents positions and orientations of
        subtomograms.
    order : int, default is 3
        Interpolation order of the rotation of subtomograms.
    scale : float, default is 1.0
        Pixel size of the reconstructed subtomograms in nm.
    output_shape : int or tuple of int, optional
        Shape of output subtomogram in pixel. This parameter is not required
        if template (or mask) image is available immediately.
    corner_safe : bool, default is False
        If true, the cubes around molecules are reconstructed larger than
        ``output_shape`` so that densities at the corners will not be lost due to
        rotation.
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
        Number of subtomograms reconstructed in one dask task. Default is 1.
    tomo_hand : int, default is 1
        Handedness of the tomogram, 1 or -1. The handedness is inverted depending on
        the positive direction of the tilt angles. If -1, the tomogram of the
        molecules is regarded as the one reconstructed from the tilt series, mirrored
        along the z axis around ``tomogram_center``.
    padding : float, default is 2.0
        Padding factor of the reconstruction. The 3D Fourier transform of each cube
        is ``padding`` times as large as the cube, which reduces the interpolation
        artifacts and the aliasing of the densities around the molecule, such as the
        neighboring molecules. Larger value is more accurate for dense molecules but
        slower.
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
        tomo_hand: int = 1,
        padding: float = 2.0,
    ) -> None:
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
        if tomo_hand not in (1, -1):
            raise ValueError(f"'tomo_hand' must be 1 or -1, got {tomo_hand!r}.")
        if padding < 1:
            raise ValueError(f"'padding' must be >= 1, got {padding!r}.")
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
        self._tomo_hand = int(tomo_hand)
        self._padding = float(padding)
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
        tomo_hand: int = 1,
        padding: float = 2.0,
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
            tomo_hand=tomo_hand,
            padding=padding,
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
    def tomo_hand(self) -> int:
        """Return the handedness of the tomogram (1 or -1)."""
        return self._tomo_hand

    @property
    def padding(self) -> float:
        """Return the padding factor of the reconstruction."""
        return self._padding

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
            tomo_hand=self.tomo_hand,
            padding=self.padding,
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

        Note that subtomograms are reconstructed in batches, so computing each
        subtomogram separately is inefficient.

        Returns
        -------
        DaskArrayList
            Each dask array returns a subtomogram on computation.
        """
        return DaskArrayList(list(self.construct_dask(output_shape, backend)))

    def construct_dask(
        self,
        output_shape: pixel | tuple[pixel, ...] | None = None,
        backend: Backend | None = None,
    ) -> da.Array:
        """Construct a dask array of subtomograms.

        Each dask task reconstructs ``batch_size`` subtomograms, so the subtomograms
        are never in memory all at once.

        Returns
        -------
        da.Array
            An 4-D array which ``arr[i]`` corresponds to the ``i``-th subtomogram.
        """
        output_shape = self._get_output_shape(output_shape)
        if self.molecules.count() == 0:
            return da.zeros((0,) + tuple(output_shape), dtype=np.float32)
        return self._plan_reconstruction(output_shape).to_dask(backend or Backend())

    def load_iter(
        self, output_shape: _ShapeType = None
    ) -> Iterator[NDArray[np.float32]]:
        """Iterate over subtomograms.

        Parameters
        ----------
        output_shape : int or tuple of int
            Shape of the output subtomograms.

        Yields
        ------
        3D array
            Subtomogram
        """
        stack = self.construct_dask(output_shape=output_shape)
        step = max(self.batch_size or 1, 64)
        for start in range(0, stack.shape[0], step):
            yield from stack[start : start + step].compute()

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

    def _plan_reconstruction(
        self, output_shape: tuple[pixel, pixel, pixel]
    ) -> _ReconstructionPlan:
        """Plan the reconstruction of the subtomograms. The tilt images are loaded."""
        import torch

        binsize = self._images.binsize_for(self.scale * _ANGSTROM_PER_NM)
        tilt_series, images = self._images.get(binsize)
        sidelength = _cube_sidelength(output_shape, self.order, self.corner_safe)
        reconstructor = self._images.reconstructor(
            binsize, self.scale * _ANGSTROM_PER_NM, sidelength, self.padding
        )
        pos, rotations = self._to_tilt_series_frame(self.molecules)
        points = (pos - self.tomogram_center) * _ANGSTROM_PER_NM
        # the projection of the molecules, including the local shifts
        with torch.no_grad():
            projected = tilt_series.project_points(
                torch.as_tensor(points, dtype=torch.float32)
            )
        # (Å from the detector center) -> (pixels of the tilt images)
        image_center = np.array(images.shape[1:]) // 2
        positions = (
            projected.cpu().numpy().astype(np.float64) / tilt_series.pixel_spacing
            + image_center
        )
        # The molecules are at the center of the cubes (sidelength // 2). Crop the
        # rotated subtomograms from the cubes, similar to `SubtomogramLoader`.
        nmole = rotations.shape[0]
        translation_0 = np.eye(4)
        translation_0[:3, 3] = sidelength // 2
        rot_mat = np.tile(np.eye(4), (nmole, 1, 1))
        rot_mat[:, :3, :3] = rotations
        translation_1 = np.eye(4)
        translation_1[:3, 3] = -(np.array(output_shape) / 2 - 0.5)
        matrices = translation_0 @ rot_mat @ translation_1
        return _ReconstructionPlan(
            reconstructor,
            positions,
            matrices.astype(np.float32),
            output_shape,
            self.order,
            self.batch_size or 1,
        )

    def _to_tilt_series_frame(
        self, mole: Molecules
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Positions (nm) and rotation matrices of molecules in the tilt series frame."""
        pos = mole.pos.astype(np.float64)
        if mole.count() == 0:
            return pos, np.zeros((0, 3, 3))
        rotations = mole.rotator.as_matrix()
        if self.tomo_hand == -1:
            # The tomogram of the molecules is the true one mirrored along the z axis
            # around the tomogram center. Convert the molecules to the true tomogram.
            pos[:, 0] = 2 * self.tomogram_center[0] - pos[:, 0]
            rotations[:, 0, :] *= -1
        return pos, rotations

    def _tilt_images(self) -> tuple[TiltSeries, NDArray[np.float32]]:
        """The tilt series and its images binned to the pixel size of this loader."""
        binsize = self._images.binsize_for(self.scale * _ANGSTROM_PER_NM)
        return self._images.get(binsize)

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

    def refine_tilt_series(
        self,
        references: Sequence[tuple[Molecules, TemplateInputType]],
        *,
        max_shift: nm = 2.0,
        grid_shape: tuple[int, int] = (1, 1),
        smoothness: float = 0.01,
        niter: int = 100,
        batch_size: int = 16,
    ) -> Self:
        """Refine the alignment of the tilt series using particles and templates.

        Similar to Bayesian polishing, the projections of the templates placed at the
        particles are compared with the tilt images to refine the shifts of each tilt
        image. If ``grid_shape`` is larger than ``(1, 1)``, the shifts are refined as
        a smooth field over each tilt image (local alignment).

        Parameters
        ----------
        references : sequence of (Molecules, template)
            Particles and their templates. The templates must be in the pixel size of
            ``scale`` and in the same contrast as the subtomograms of this loader.
            Particles of different kinds can be given with different templates.
        max_shift : float, default is 2.0
            Maximum shift of the tilt images in nm.
        grid_shape : (int, int), default is (1, 1)
            Number of control points of the local shifts along y and x of the tilt
            images. The control points are evenly placed from edge to edge of the
            images. ``(1, 1)`` means a uniform shift for each tilt image.
        smoothness : float, default is 0.01
            Weight of the smoothness prior of the local shifts. The penalty is the sum
            of squared differences (nm^2) between the shifts of neighboring control
            points, compared with the sum of normalized cross correlation over
            particles. Larger value is more robust against noise and fewer particles.
        niter : int, default is 100
            Number of iterations to optimize the local shifts.
        batch_size : int, default is 16
            Number of particles processed at once.

        Returns
        -------
        PseudoSubtomogramLoader
            A new loader with the refined tilt series. ``sample_translations`` of the
            tilt series are updated, and the local shifts are set to
            ``local_shifts_2d`` if ``grid_shape`` is larger than ``(1, 1)``.
        """
        from acryo.loader._tilt_refine import ParticleSet, refine_tilt_series

        particle_sets: list[ParticleSet] = []
        for mole, template in references:
            pos, rotations = self._to_tilt_series_frame(mole)
            points = ((pos - self.tomogram_center) * _ANGSTROM_PER_NM).astype(
                np.float32
            )
            particle_sets.append(
                ParticleSet(points, rotations, self.normalize_template(template))
            )
        import torch

        tilt_series, images = self._tilt_images()
        with warnings.catch_warnings():
            # The cached images are read-only, but they are not modified, so they are
            # not copied.
            warnings.filterwarnings("ignore", message=".*not writable.*")
            images_tensor = torch.as_tensor(images, device=tilt_series.device)
        refined = refine_tilt_series(
            tilt_series,
            images_tensor,
            particle_sets,
            pixel_spacing=self.scale * _ANGSTROM_PER_NM,
            max_shift=max_shift * _ANGSTROM_PER_NM,
            grid_shape=grid_shape,
            smoothness=smoothness,
            niter=niter,
            batch_size=batch_size,
        )
        # the geometry is in Å, so it does not depend on the binning of the images
        tilt_series = copy.copy(self.tilt_series)
        tilt_series.sample_translations = refined.sample_translations
        tilt_series.local_shifts_2d = refined.local_shifts_2d
        out = self.replace()
        out._images = self._images.with_tilt_series(tilt_series)
        return out

    def _default_align_kwargs(self) -> dict[str, Any]:
        """Return default keyword arguments for alignment."""
        return {"tilt": self.tilt_model}


class _TiltSeriesImages:
    """Tilt series and its images that are loaded only once on demand.

    The images are read from the file tilt by tilt, preprocessed at the original pixel
    spacing and binned by an integer factor. The images of each binning factor are
    cached in memory. The reconstructors of the latest geometries of cubes are also
    cached.
    """

    _MAX_RECONSTRUCTORS = 2

    def __init__(
        self,
        tilt_series: TiltSeries,
        preprocess: bool,
        preprocess_kwargs: dict[str, Any] | None,
    ):
        self.tilt_series = tilt_series
        self.preprocess = preprocess
        self.preprocess_kwargs = dict(preprocess_kwargs or {})
        self._images: dict[int, NDArray[np.float32]] = {}
        self._lock = threading.Lock()
        # reconstructors depend on the geometry, so they are not shared with the
        # instances of other tilt series
        self._reconstructors: dict[
            tuple[int, float, int, float], DirectFourierReconstructor
        ] = {}

    def binsize_for(self, pixel_spacing: float) -> int:
        """Largest binning factor whose pixel spacing does not exceed the given one."""
        ratio = pixel_spacing / self.tilt_series.pixel_spacing
        return max(math.floor(ratio + 1e-6), 1)

    def get(self, binsize: int = 1) -> tuple[TiltSeries, NDArray[np.float32]]:
        """Get the tilt series and the (preprocessed) tilt images binned by binsize.

        The returned tilt series is a shallow copy with the binned pixel spacing.
        """
        with self._lock:
            if binsize not in self._images:
                self._images[binsize] = self._load(binsize)
            images = self._images[binsize]
        if binsize == 1:
            return self.tilt_series, images
        tilt_series = copy.copy(self.tilt_series)
        tilt_series.pixel_spacing = self.tilt_series.pixel_spacing * binsize
        return tilt_series, images

    def reconstructor(
        self, binsize: int, output_spacing: float, sidelength: int, padding: float
    ) -> DirectFourierReconstructor:
        """Get the reconstructor of cubes from the tilt images binned by binsize."""
        key = (binsize, round(output_spacing, 6), sidelength, padding)
        if (out := self._reconstructors.get(key)) is not None:
            return out
        tilt_series, images = self.get(binsize)
        # from the detector to the tomogram space
        matrices = tilt_series.projection_matrices @ tilt_series.tomo2sample
        rotations = np.linalg.pinv(
            matrices[:, :3, :3].detach().cpu().numpy().astype(np.float64)
        )
        out = DirectFourierReconstructor(
            images,
            rotations,
            input_spacing=tilt_series.pixel_spacing,
            output_spacing=output_spacing,
            sidelength=sidelength,
            padding=padding,
        )
        with self._lock:
            self._reconstructors[key] = out
            while len(self._reconstructors) > self._MAX_RECONSTRUCTORS:
                self._reconstructors.pop(next(iter(self._reconstructors)))
        return out

    def with_tilt_series(self, tilt_series: TiltSeries) -> _TiltSeriesImages:
        """Return a new instance of another geometry for the same tilt images."""
        out = _TiltSeriesImages(tilt_series, self.preprocess, self.preprocess_kwargs)
        out._images = self._images
        out._lock = self._lock
        return out

    def _load(self, binsize: int) -> NDArray[np.float32]:
        import torch

        tilt_series = self.tilt_series
        device = tilt_series.device
        stack = _MrcImageStack(tilt_series.image_path)
        if tilt_series.image_indices is None:
            indices = list(range(stack.shape[0]))
        else:
            indices = tilt_series.image_indices.tolist()
        out: NDArray[np.float32] | None = None
        preprocessor: _TiltPreprocessor | None = None
        for t, raw in enumerate(stack.iter_images(indices)):
            image = torch.as_tensor(raw, device=device)
            if self.preprocess:
                if preprocessor is None:
                    preprocessor = _TiltPreprocessor(
                        tuple(image.shape), device, **self.preprocess_kwargs
                    )
                image = preprocessor(image)
            image = _fourier_bin(image, binsize).cpu().numpy()
            if out is None:
                out = np.empty((len(indices),) + image.shape, dtype=np.float32)
            out[t] = image
        assert out is not None
        # the cache is shared by the loaders
        out.setflags(write=False)
        return out


class _MrcImageStack:
    """An MRC file of a stack of 2D images, such as a tilt series.

    Images are read one by one from the file, so the whole stack is never read nor
    memory-mapped.
    """

    def __init__(self, path):
        import mrcfile
        from mrcfile.utils import data_dtype_from_header

        with mrcfile.open(path, mode="r", header_only=True, permissive=True) as mrc:
            header = mrc.header
            self.shape = (int(header.nz), int(header.ny), int(header.nx))
            self.dtype = data_dtype_from_header(header)
            # the data block follows the main header and the extended header
            self.offset = header.nbytes + int(header.nsymbt)
        self.path = path

    def iter_images(self, indices: Iterable[int]) -> Iterator[NDArray[np.float32]]:
        """Iterate over the images of given indices as float32 arrays."""
        nimages, ny, nx = self.shape
        with open(self.path, "rb") as f:
            for index in indices:
                if not -nimages <= index < nimages:
                    raise IndexError(
                        f"Index {index} is out of range of {nimages} images in "
                        f"{self.path}."
                    )
                f.seek(self.offset + (index % nimages) * ny * nx * self.dtype.itemsize)
                image = np.fromfile(f, dtype=self.dtype, count=ny * nx)
                if image.size != ny * nx:
                    raise ValueError(f"The data block of {self.path} is truncated.")
                yield image.reshape(ny, nx).astype(np.float32)


class _TiltPreprocessor:
    """Preprocess tilt images one by one.

    This is the same as ``torch_tilt_series.preprocess_tilt_series_images``, but the
    plane fitting and the filters are computed only once for all the tilt images.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        device,
        low: float = 0.0,
        high: float = 0.5,
        falloff: float = 0.0,
        bandpass_padding: int = 128,
        subtract_background: bool = True,
        normalize: bool = True,
    ):
        import torch
        from torch_fourier_filter.bandpass import bandpass_filter
        from torch_grid_utils.shapes_2d import rectangle

        h, w = shape
        self.shape = shape
        self.subtract_background = subtract_background
        self.normalize = normalize
        if subtract_background:
            y, x = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing="ij",
            )
            ones = torch.ones(h * w, device=device, dtype=torch.float32)
            self.design = torch.stack([y.reshape(-1), x.reshape(-1), ones], dim=-1)
            self.pinv = torch.linalg.pinv(self.design)
        self.pad = pad = max(0, min(bandpass_padding, h - 1, w - 1))
        padded_shape = (h + 2 * pad, w + 2 * pad)
        self.mask = rectangle(
            dimensions=(h, w),
            image_shape=padded_shape,
            smoothing_radius=pad,
            device=device,
        )
        self.filter = bandpass_filter(
            low=low,
            high=high,
            falloff=falloff,
            image_shape=padded_shape,
            rfft=True,
            fftshift=False,
            device=device,
        )

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        import torch
        import torch.nn.functional as F
        from torch_tilt_series.utils import normalize_on_central_crop

        h, w = self.shape
        if self.subtract_background:
            coeffs = image.reshape(1, h * w) @ self.pinv.T
            image = image - (coeffs @ self.design.T).reshape(h, w)
        pad = self.pad
        padded = F.pad(image[None], [pad, pad, pad, pad], mode="reflect")
        padded = padded * self.mask
        padded = torch.fft.irfft2(
            torch.fft.rfft2(padded) * self.filter, s=padded.shape[-2:]
        )
        image = padded[0, pad : pad + h, pad : pad + w]
        if self.normalize:
            image = normalize_on_central_crop(image)
        return image


def _fourier_bin(image: torch.Tensor, binsize: int) -> torch.Tensor:
    """Bin a 2D image by Fourier cropping, keeping the position of the image center.

    The image center (``size // 2``) is the origin of the detector, so the image is
    first cropped symmetrically around the center to a multiple of ``2 * binsize``.
    The pixel spacing of the output is exactly ``binsize`` times larger.
    """
    from torch_fourier_rescale import fourier_rescale_2d

    if binsize == 1:
        return image
    slices: list[slice] = []
    target_shape: list[int] = []
    for size in image.shape[-2:]:
        cropped = size - size % (2 * binsize)
        start = size // 2 - cropped // 2
        slices.append(slice(start, start + cropped))
        target_shape.append(cropped // binsize)
    binned, _ = fourier_rescale_2d(
        image[..., slices[0], slices[1]], target_shape=tuple(target_shape)
    )
    return binned


def _cube_sidelength(
    output_shape: tuple[int, int, int], order: int, corner_safe: bool
) -> int:
    """Sidelength of the cubes to crop the subtomograms with the interpolation margin.

    The molecule is at the center of the cube (``sidelength // 2``).
    """
    half = (np.asarray(output_shape, dtype=np.float64) - 1) / 2
    radius = float(np.linalg.norm(half)) if corner_safe else float(half.max())
    return 2 * math.ceil(radius + order) + 1


class _ReconstructionPlan:
    """Reconstruct the subtomograms in batches."""

    def __init__(
        self,
        reconstructor: DirectFourierReconstructor,
        positions: NDArray[np.float64],
        matrices: NDArray[np.float32],
        output_shape: tuple[int, int, int],
        order: int,
        batch_size: int,
    ):
        self.reconstructor = reconstructor
        self.positions = positions  # (N, T, 2) positions on the tilt images
        self.matrices = matrices  # (N, 4, 4) from the subtomograms to the cubes
        self.output_shape = tuple(output_shape)
        self.order = order
        self.batch_size = batch_size

    def to_dask(self, backend: Backend) -> da.Array:
        """Construct the dask array of the subtomograms."""
        output_shape = self.output_shape
        name = f"pseudo-subvolumes-{uuid.uuid4().hex}"
        zeros = (0,) * len(output_shape)
        task = _ReconstructionTask(
            self.reconstructor, output_shape, self.order, backend
        )
        layer = {}
        sizes = []
        nmole = self.positions.shape[0]
        for ib, start in enumerate(range(0, nmole, self.batch_size)):
            sl = slice(start, start + self.batch_size)
            layer[(name, ib, *zeros)] = (task, self.positions[sl], self.matrices[sl])
            sizes.append(self.positions[sl].shape[0])
        graph = HighLevelGraph.from_collections(name, layer, dependencies=[])
        chunks = (tuple(sizes),) + tuple((s,) for s in output_shape)
        meta = np.empty((0,) * (len(output_shape) + 1), dtype=np.float32)
        return da.Array(graph, name, chunks, meta=meta)


class _ReconstructionTask:
    """Reconstruct a batch of subtomograms in a dask task."""

    def __init__(
        self,
        reconstructor: DirectFourierReconstructor,
        output_shape: tuple[int, int, int],
        order: int,
        backend: Backend,
    ):
        self.reconstructor = reconstructor
        self.output_shape = output_shape
        self.order = order
        self.backend = backend

    def __call__(
        self, positions: NDArray[np.float64], matrices: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        xp = self.backend
        cubes = self.reconstructor(positions)
        # the tilt images have dark densities on a light background
        cubes *= -1
        subvolumes = [
            xp.rotated_crop(
                xp.asarray(cube),
                matrix,
                shape=self.output_shape,
                order=self.order,
                cval=xp.mean,
            )
            for cube, matrix in zip(cubes, matrices)
        ]
        return xp.stack(subvolumes, axis=0)


_PAD_FACTOR = 2.0


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
