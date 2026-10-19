# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from contextlib import contextmanager
import functools
import math
import os
from pathlib import Path
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Generator,
    Sequence,
    SupportsIndex,
)
import uuid

import numpy as np
from numpy.typing import DTypeLike, NDArray
import dask
from dask import array as da
from dask.highlevelgraph import HighLevelGraph

from acryo._types import nm, pixel
from acryo._dask import DaskArrayList
from acryo.backend import Backend
from acryo.loader._base import LoaderBase, Unset, _ShapeType, TemplateInputType
from acryo.molecules import Molecules
from acryo.ctf import CTFCorrection, PatchCTFCorrection
from acryo.projection import (
    DirectFourierReconstructor,
    TiltImagePreprocessor,
    TiltSeries,
    fourier_bin,
    reconstruct_tomogram,
)
from acryo.projection._io import TiltImageSource
from acryo.tilt import TiltSeriesModel, single_axis

if TYPE_CHECKING:
    from typing_extensions import Self

# Memory budget (bytes) of the tilt images being preprocessed in parallel.
_LOAD_MEMORY_BUDGET = 2e9
# Only one tilt series is preprocessed at a time, so that the memory budget is kept
# when many tilt series are loaded in parallel dask tasks.
_LOAD_LOCK = threading.Lock()


class BackProjectionLoader(LoaderBase):
    """A loader that reconstructs subtomograms directly from a tilt series.

    No tomogram is needed. Every time subtomograms are requested, they are
    reconstructed from the tilt series at the pixel size ``scale``, which is the pixel
    size of the tilt images (``tilt_series.scale``) times ``binsize``, by direct
    Fourier inversion (insertion of the central slices into the 3D Fourier
    transform), as in ``torch-reconstruct-tomogram``. A cube aligned to the tomogram
    axes is reconstructed around each molecule, and the subtomogram is cropped from
    the cube in the same way as ``SubtomogramLoader``. Subtomograms are reconstructed
    lazily in dask tasks. The CTF of the tilt images is corrected in preprocessing if
    the tilt series has the CTF parameters (see ``preprocess``). Reading the tilt
    images requires ``mrcfile``.

    Parameters
    ----------
    tilt_series : acryo.projection.TiltSeries
        Tilt series with the alignment parameters. ``image_path`` and ``scale`` must
        be set. The tilt images are read from the file tilt by tilt, preprocessed and
        binned by the largest integer factor whose pixel size does not exceed
        ``scale``, so the raw tilt series is never in memory all at once. For binning
        by 4 or more, the images are binned partly in real space before preprocessing
        (see ``TiltImagePreprocessor``). Each tilt image is read and preprocessed in a
        task of a dask graph, and the tasks run in parallel threads, as many as fit in
        about 2 GB of memory. Only one tilt series is preprocessed at a time. The tilt
        images are loaded in the computation of the dask array of
        :meth:`construct_dask`, and released after that, so the tilt series of many
        loaders (such as in a
        ``BatchLoader``) are not all in memory at once. Use :meth:`cache_tilt_images`
        to reuse them in repeated computations.
    molecules : Molecules
        Molecules object that represents positions and orientations of
        subtomograms.
    order : int, default is 3
        Interpolation order of the rotation of subtomograms.
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
        by :meth:`reconstruct_tomogram` (or ``torch-reconstruct-tomogram``), this is
        ``(D // 2 * s, H // 2 * s, W // 2 * s)``.
    tilt_model : TiltSeriesModel, optional
        Tilt series model used for alignment. By default, a single-axis model around
        the y-axis is created from the tilt angles of the tilt series (and
        ``tomo_hand``).
    preprocess : bool or TiltImagePreprocessor, default is True
        Preprocessing of the tilt images. If true, the tilt images are preprocessed by
        the default ``TiltImagePreprocessor()``. Give a ``TiltImagePreprocessor`` to
        change the parameters or to add filters (see
        ``TiltImagePreprocessor.with_filters``). If the tilt series has the CTF
        parameters (``tilt_series.ctf``, which is read by ``TiltSeries.from_*`` by
        default), the CTF correction of each patch (``tilt_series.ctf_correction()``,
        whose defocus gradient is of the standard handedness) is added to the
        filters, unless the preprocessor already has a CTF correction
        (``acryo.ctf.CTFCorrection`` or ``acryo.ctf.PatchCTFCorrection``). If false, the tilt
        images are not preprocessed (and the CTF is not corrected).
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
    binsize : float, default is 1.0
        Binning factor of the subtomograms relative to the tilt images. The pixel size
        of the subtomograms (``scale``) is ``tilt_series.scale * binsize``. It does not
        have to be an integer. :meth:`binning` returns a loader of a larger binsize,
        and ``replace(binsize=...)`` returns a loader of the given binsize.
    """

    def __init__(
        self,
        tilt_series: TiltSeries,
        molecules: Molecules | None = None,
        order: int = 3,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
        tomogram_center: tuple[nm, nm, nm] = (0.0, 0.0, 0.0),
        tilt_model: TiltSeriesModel | None = None,
        preprocess: bool | TiltImagePreprocessor = True,
        batch_size: int | None = None,
        tomo_hand: int = 1,
        padding: float = 2.0,
        binsize: float = 1.0,
    ) -> None:
        tilt_series = _as_tilt_series(tilt_series)
        if tilt_series.image_path is None:
            raise ValueError("'image_path' of the tilt series is not set.")
        if tilt_series.scale is None:
            raise ValueError("'scale' of the tilt series is not set.")
        if isinstance(preprocess, TiltImagePreprocessor):
            preprocessor = preprocess
        elif isinstance(preprocess, bool):
            preprocessor = TiltImagePreprocessor() if preprocess else None
        else:
            raise TypeError(
                "'preprocess' must be a bool or a TiltImagePreprocessor, got "
                f"{type(preprocess)}."
            )
        if molecules is None:
            molecules = Molecules.empty()
        elif not isinstance(molecules, Molecules):
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
        if binsize <= 0:
            raise ValueError(f"'binsize' must be positive, got {binsize!r}.")
        # the default tilt model is created again if the tilts are excluded
        self._default_tilt_model = tilt_model is None
        if tilt_model is None:
            tilt_model = _default_tilt_model(tilt_series, tomo_hand)
        self._images = _TiltSeriesImages(tilt_series, preprocessor)
        self._molecules = molecules
        self._corner_safe = corner_safe
        self._tomogram_center = _center
        self._tilt_model = tilt_model
        self._batch_size = None if batch_size is None else int(batch_size)
        self._tomo_hand = int(tomo_hand)
        self._padding = float(padding)
        self._binsize = float(binsize)
        super().__init__(
            order=order,
            scale=tilt_series.scale * self._binsize,
            output_shape=output_shape,
        )

    def __repr__(self) -> str:
        mole_repr = repr(self.molecules)
        return (
            f"{self.__class__.__name__}(tilt_series=<{self.tilt_series.ntilts} tilts>, "
            f"molecules={mole_repr}, output_shape={self.output_shape}, "
            f"order={self.order}, binsize={self.binsize:g}, scale={self.scale:.4f})"
        )

    @property
    def tilt_series(self) -> TiltSeries:
        """Return the tilt series."""
        return self._images.tilt_series

    @property
    def preprocessor(self) -> TiltImagePreprocessor | None:
        """Return the preprocessor of the tilt images, including the CTF correction."""
        return self._images.effective_preprocessor()

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

    @property
    def binsize(self) -> float:
        """Return the binning factor of the subtomograms relative to the tilt images."""
        return self._binsize

    @contextmanager
    def cache_tilt_images(self) -> Generator[Self, None, None]:
        """Context manager to cache the preprocessed tilt images in memory.

        By default, the tilt images are read, preprocessed and binned every time
        subtomograms are requested, and released after that. In this context, the
        preprocessed tilt images (of each binning factor) and the reconstructors are
        cached, and reused by this loader and the loaders created from it, such as the
        results of :meth:`replace`, :meth:`binning`, :meth:`align`,
        :meth:`with_tilt_series`, :meth:`refine_tilt_series` and
        :meth:`exclude_tilts`. The cache is released when the outermost context exits.
        Lazy arrays created in the context can still be computed after that.

        Examples
        --------
        >>> with loader.cache_tilt_images():
        ...     aligned = loader.align(template, max_shifts=1.0)
        ...     avg = aligned.average()
        """
        cache = self._images.cache
        cache.enter()
        try:
            yield self
        finally:
            cache.exit()

    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset | None = None,
        order: int | None = None,
        scale: float | None = None,
        *,
        binsize: float | None = None,
    ) -> Self:
        """Return a new instance with different parameter(s).

        The pixel size of the subtomograms is changed by ``binsize``, the binning
        factor relative to the tilt images. ``scale`` (the pixel size in nm) is also
        accepted for the compatibility with other loaders, such as ``groupby``, and is
        converted to ``binsize``.
        """
        if molecules is None:
            molecules = self.molecules
        if output_shape is None:
            output_shape = self.output_shape
        if order is None:
            order = self.order
        if scale is not None:
            if binsize is not None:
                raise TypeError("Cannot specify both 'scale' and 'binsize'.")
            binsize = scale / self._images.scale
        elif binsize is None:
            binsize = self.binsize
        out = self.__class__(
            self.tilt_series,
            molecules=molecules,
            order=order,
            output_shape=output_shape,
            corner_safe=self.corner_safe,
            tomogram_center=tuple(self.tomogram_center),
            tilt_model=self.tilt_model,
            batch_size=self.batch_size,
            tomo_hand=self.tomo_hand,
            padding=self.padding,
            binsize=binsize,
        )
        out._images = self._images  # share the loaded tilt images
        out._default_tilt_model = self._default_tilt_model
        return out

    def with_tilt_series(self, tilt_series: TiltSeries) -> Self:
        """Return a new instance with another tilt series, such as a refined one.

        If the tilt series has the same tilt images (``image_path``,
        ``image_indices`` and ``scale``), the preprocessed images and their cache are
        shared with this loader, so updating the alignment is cheap.
        """
        tilt_series = _as_tilt_series(tilt_series)
        out = self.replace()
        if _same_tilt_images(self.tilt_series, tilt_series):
            out._images = self._images.with_tilt_series(tilt_series)
        else:
            out._images = _TiltSeriesImages(tilt_series, self._images.preprocessor)
        return out

    def exclude_tilts(self, indices: SupportsIndex | Iterable[SupportsIndex]) -> Self:
        """Return a new instance without the given tilts.

        Subtomograms are reconstructed from the rest of the tilt images, and the images
        of the excluded tilts are not read. The tilt series and the preprocessor are
        converted by their ``select_tilts`` methods, so the filters of each tilt, such
        as the CTF correction, are kept. If ``tilt_model`` was not given, it is created
        again from the remaining tilt angles. The cache of the tilt images (see
        :meth:`cache_tilt_images`) is shared with this loader.

        Parameters
        ----------
        indices : int or sequence of int
            Indices of the tilts to exclude in the tilt series of this loader (such as
            ``tilt_series.tilt_angles[i]``). They are not the indices of the images in
            the file, which differ if the tilt series is read with excluded views
            (such as by :meth:`TiltSeries.from_etomo_directory`).

        Examples
        --------
        >>> loader.exclude_tilts([0, 1])  # exclude the first two tilts
        >>> angles = loader.tilt_series.tilt_angles
        >>> loader.exclude_tilts(np.flatnonzero(np.abs(angles) > 50))
        """
        ntilts = self.tilt_series.ntilts
        excluded = np.atleast_1d(np.asarray(indices))
        if excluded.size == 0:
            return self.replace()
        if excluded.ndim != 1 or not np.issubdtype(excluded.dtype, np.integer):
            raise TypeError(
                f"'indices' must be int or a sequence of int, got {indices!r}."
            )
        excluded = np.arange(ntilts)[excluded]  # IndexError if out of range
        keep = np.setdiff1d(np.arange(ntilts), excluded)
        if keep.size == 0:
            raise ValueError("Cannot exclude all the tilts.")
        out = self.replace()
        out._images = self._images.select_tilts(keep)
        if self._default_tilt_model:
            out._tilt_model = _default_tilt_model(out.tilt_series, self.tomo_hand)
        return out

    def binning(self, binsize: float = 2, *, compute: bool = True) -> Self:
        """Return a new instance with larger pixel size.

        Subtomograms are reconstructed at the binned pixel size, so ``compute`` has
        no effect. It is only for the compatibility with ``SubtomogramLoader``.
        """
        if binsize <= 0:
            raise ValueError(f"'binsize' must be positive, got {binsize!r}.")
        if binsize == 1:
            return self.copy()
        return self.replace(binsize=self.binsize * binsize)

    def construct_loading_tasks(
        self,
        output_shape: _ShapeType = None,
        backend: Backend | None = None,
    ) -> DaskArrayList:
        """Construct a list of subtomogram lazy loader.

        Note that subtomograms are reconstructed in batches, so computing each
        subtomogram separately is inefficient. Unlike :meth:`construct_dask`, the tilt
        images are loaded here, so that they are not loaded again for each subtomogram.

        Returns
        -------
        DaskArrayList
            Each dask array returns a subtomogram on computation.
        """
        stack = self._construct_dask(output_shape, backend, lazy=False)
        return DaskArrayList(list(stack))

    def construct_dask(
        self,
        output_shape: pixel | tuple[pixel, ...] | None = None,
        backend: Backend | None = None,
    ) -> da.Array:
        """Construct a dask array of subtomograms.

        Each dask task reconstructs ``batch_size`` subtomograms, so the subtomograms
        are never in memory all at once. The tilt images are loaded by the first task
        in the computation, and released after all the tasks, unless they are cached
        by :meth:`cache_tilt_images`. Note that each computation of the whole array
        loads the tilt images again.

        Returns
        -------
        da.Array
            An 4-D array which ``arr[i]`` corresponds to the ``i``-th subtomogram.
        """
        return self._construct_dask(output_shape, backend, lazy=True)

    def _construct_dask(
        self,
        output_shape: pixel | tuple[pixel, ...] | None,
        backend: Backend | None,
        lazy: bool,
    ) -> da.Array:
        output_shape = self._get_output_shape(output_shape)
        if self.molecules.count() == 0:
            return da.zeros((0,) + tuple(output_shape), dtype=np.float32)
        # The cached tilt images are captured by the dask array, so that it can be
        # computed after the cache is released.
        lazy = lazy and not self._images.cache.enabled
        plan = self._plan_reconstruction(output_shape)
        return plan.to_dask(backend or Backend(), lazy=lazy)

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
        # the tilt images are loaded once for all the steps
        stack = self._construct_dask(output_shape, None, lazy=False)
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
        """Plan the reconstruction without loading the tilt images."""
        binsize = self._images.binsize_for(self.binsize)
        sidelength = _cube_sidelength(output_shape, self.order, self.corner_safe)
        get_reconstructor = functools.partial(
            self._images.reconstructor, binsize, self.scale, sidelength, self.padding
        )
        tilt_series = self._images.binned_tilt_series(binsize)
        pos, rotations = self._to_tilt_series_frame(self.molecules)
        # the projection of the molecules, including the local shifts
        projected = tilt_series.project_points(pos - self.tomogram_center)
        # (nm from the detector center) -> (pixels from the center of the tilt images)
        offsets = projected / tilt_series.scale
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
            get_reconstructor,
            offsets,
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
        binsize = self._images.binsize_for(self.binsize)
        return self._images.get(binsize)

    def reconstruct_tomogram(
        self,
        shape: tuple[pixel, pixel, pixel],
        *,
        patch_size: pixel = 64,
        blend_margin: pixel | None = None,
        save_path: str | Path | None = None,
        overwrite: bool = False,
        num_workers: int | None = None,
        dtype: DTypeLike = np.float16,
    ) -> NDArray[np.floating]:
        """Reconstruct the tomogram from the tilt series.

        The tomogram is reconstructed at the pixel size of ``scale`` from the
        preprocessed tilt images, by direct Fourier inversion of patches (see
        :func:`acryo.projection.reconstruct_tomogram`). The tomogram is centered at
        ``tomogram_center``, that is, the voxel ``(i, j, k)`` is at
        ``tomogram_center + ((i, j, k) - shape // 2) * scale`` nm in the coordinate
        system of the molecules. If ``tomogram_center`` is ``shape // 2 * scale``,
        ``SubtomogramLoader(tomogram, loader.molecules, scale=loader.scale)`` loads
        the same subtomograms as this loader. If ``tomo_hand`` is -1, the tomogram is
        mirrored along the z axis around the center. The contrast of the tilt images
        is kept (not inverted).

        Parameters
        ----------
        shape : int or tuple of int
            Shape of the tomogram in pixel.
        patch_size : int, default is 64
            Distance between the centers of the reconstructed patches in pixel.
        blend_margin : int, optional
            Margin of each patch in pixel, which is blended with the neighboring
            patches. Default is ``patch_size // 4``.
        save_path : str or Path, optional
            If given, the tomogram is written to this MRC file, which is allocated
            beforehand, and the tomogram is never in memory all at once.
        overwrite : bool, default is False
            Whether to overwrite the existing file at ``save_path``.
        num_workers : int, optional
            Number of threads to reconstruct the patches. Default is the number of
            the CPUs. Each thread needs memory for a padded patch, so a smaller value
            uses less memory.
        dtype : float16 or float32, default is float16
            Data type of the tomogram. If ``save_path`` is given, the MRC file is of
            mode 12 (float16) or mode 2 (float32).

        Returns
        -------
        np.ndarray
            The reconstructed tomogram. If ``save_path`` is given, a read-only memory
            map of the file.
        """
        binsize = self._images.binsize_for(self.binsize)
        tilt_series, images = self._images.get(binsize)
        if self.tomo_hand == -1:
            # The tomogram of the molecules is mirrored along the z axis.
            tilt_series = tilt_series.replace(
                levelled2tomo=np.diag([-1.0, 1.0, 1.0, 1.0]) @ tilt_series.levelled2tomo
            )
        nz, ny, nx = shape
        if nz < 0:
            raise ValueError(f"'nz' must be non-negative, got {nz}.")
        if ny < 0:
            ny = images.shape[1]
        if nx < 0:
            nx = images.shape[2]
        return reconstruct_tomogram(
            tilt_series,
            images,
            (nz, ny, nx),
            scale=self.scale,
            patch_size=patch_size,
            blend_margin=blend_margin,
            padding=self.padding,
            save_path=save_path,
            overwrite=overwrite,
            num_workers=num_workers,
            dtype=dtype,
        )

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
            Maximum number of iterations to optimize the local shifts (L-BFGS-B).
        batch_size : int, default is 16
            Number of particles processed at once.

        Returns
        -------
        BackProjectionLoader
            A new loader with the refined tilt series. ``sample_translations`` of the
            tilt series are updated, and the local shifts are set to
            ``local_shifts_2d`` if ``grid_shape`` is larger than ``(1, 1)``.
        """
        from acryo.loader._tilt_refine import ParticleSet, refine_tilt_series

        particle_sets: list[ParticleSet] = []
        for mole, template in references:
            pos, rotations = self._to_tilt_series_frame(mole)
            particle_sets.append(
                ParticleSet(
                    pos - self.tomogram_center,
                    rotations,
                    self.normalize_template(template),
                )
            )
        tilt_series, images = self._tilt_images()
        refined = refine_tilt_series(
            tilt_series,
            images,
            particle_sets,
            scale=self.scale,
            max_shift=max_shift,
            grid_shape=grid_shape,
            smoothness=smoothness,
            niter=niter,
            batch_size=batch_size,
        )
        # the geometry is in nm, so it does not depend on the binning of the images
        tilt_series = self.tilt_series.replace(
            sample_translations=refined.sample_translations,
            local_shifts_2d=refined.local_shifts_2d,
        )
        return self.with_tilt_series(tilt_series)

    def _default_align_kwargs(self) -> dict[str, Any]:
        """Return default keyword arguments for alignment."""
        return {"tilt": self.tilt_model}


class _ImageCache:
    """Cache of the preprocessed tilt images, which is enabled only in the context.

    The cache is shared by the loaders created from the same loader. The images and
    the reconstructors are released when the outermost context exits.
    """

    _MAX_RECONSTRUCTORS = 2

    def __init__(self):
        self.lock = threading.RLock()
        self._depth = 0
        # (binsize, selection of the tilts) -> images
        self.images: dict[tuple[int, bytes | None], NDArray[np.float32]] = {}
        self.reconstructors: dict[tuple[Any, ...], DirectFourierReconstructor] = {}

    @property
    def enabled(self) -> bool:
        """Whether the cache is enabled."""
        return self._depth > 0

    def enter(self) -> None:
        with self.lock:
            self._depth += 1

    def exit(self) -> None:
        with self.lock:
            self._depth -= 1
            if self._depth == 0:
                self.images.clear()
                self.reconstructors.clear()

    def add_reconstructor(
        self, key: tuple[Any, ...], reconstructor: DirectFourierReconstructor
    ) -> None:
        self.reconstructors[key] = reconstructor
        while len(self.reconstructors) > self._MAX_RECONSTRUCTORS:
            self.reconstructors.pop(next(iter(self.reconstructors)))


class _TiltSeriesImages:
    """Tilt series and its preprocessed images.

    The images are read from the file, and preprocessed and binned by an integer factor
    in the tasks of a dask graph (one for each tilt image). They are cached only while
    the cache is enabled.
    """

    def __init__(
        self,
        tilt_series: TiltSeries,
        preprocessor: TiltImagePreprocessor | None,
        cache: _ImageCache | None = None,
        selection: NDArray[np.intp] | None = None,
    ):
        self.tilt_series = tilt_series
        self.preprocessor = preprocessor
        self.cache = cache if cache is not None else _ImageCache()
        # Indices of the tilts in the images that created the cache, if some tilts are
        # excluded. The images are cropped from the cached images of all the tilts.
        self.selection = selection

    def _cache_key(self, binsize: int) -> tuple[int, bytes | None]:
        sel = self.selection
        return binsize, None if sel is None else sel.tobytes()

    def effective_preprocessor(self) -> TiltImagePreprocessor | None:
        """The preprocessor with the CTF correction of the tilt series, if any."""
        preprocessor = self.preprocessor
        if (
            preprocessor is None
            or self.tilt_series.ctf is None
            or any(
                isinstance(f, (CTFCorrection, PatchCTFCorrection))
                for f in preprocessor.filters
            )
        ):
            return preprocessor
        return preprocessor.with_filters(self.tilt_series.ctf_correction())

    @property
    def scale(self) -> nm:
        """Pixel size of the raw tilt images in nm."""
        assert self.tilt_series.scale is not None
        return self.tilt_series.scale

    def binsize_for(self, binsize: float) -> int:
        """Binning factor of the tilt images for the binsize of the subtomograms.

        The images are binned by the largest integer that does not exceed ``binsize``,
        so that the subtomograms are not affected by binning. The images are cached
        for each integer binning factor.
        """
        return max(math.floor(binsize + 1e-6), 1)

    def binned_tilt_series(self, binsize: int) -> TiltSeries:
        """The tilt series whose scale is the binned pixel size."""
        if binsize == 1:
            return self.tilt_series
        return self.tilt_series.replace(scale=self.scale * binsize)

    def get(self, binsize: int = 1) -> tuple[TiltSeries, NDArray[np.float32]]:
        """Get the tilt series and the (preprocessed) tilt images binned by binsize.

        The scale of the returned tilt series is the binned pixel size.
        """
        cache = self.cache
        with cache.lock:
            key = self._cache_key(binsize)
            images = cache.images.get(key)
            if images is None and self.selection is not None:
                all_images = cache.images.get((binsize, None))
                if all_images is not None:
                    images = all_images[self.selection]
                    images.setflags(write=False)
            if images is None:
                images = self._load(binsize)
                if cache.enabled:
                    cache.images[key] = images
        return self.binned_tilt_series(binsize), images

    def reconstructor(
        self, binsize: int, output_scale: nm, sidelength: int, padding: float
    ) -> DirectFourierReconstructor:
        """Get the reconstructor of cubes from the tilt images binned by binsize."""
        rotations = self.tilt_series.detector2tomo
        key = (
            self._cache_key(binsize),
            round(output_scale, 8),
            sidelength,
            padding,
            rotations.tobytes(),
        )
        cache = self.cache
        with cache.lock:
            if (out := cache.reconstructors.get(key)) is not None:
                return out
            _, images = self.get(binsize)
            out = DirectFourierReconstructor(
                images,
                rotations,
                input_spacing=self.scale * binsize,
                output_spacing=output_scale,
                sidelength=sidelength,
                padding=padding,
            )
            if cache.enabled:
                cache.add_reconstructor(key, out)
        return out

    def with_tilt_series(self, tilt_series: TiltSeries) -> _TiltSeriesImages:
        """Return a new instance of another geometry for the same tilt images."""
        return _TiltSeriesImages(
            tilt_series, self.preprocessor, cache=self.cache, selection=self.selection
        )

    def select_tilts(self, indices: NDArray[np.intp]) -> _TiltSeriesImages:
        """Return a new instance of the selected tilts, which shares the cache."""
        preprocessor = self.preprocessor
        if preprocessor is not None:
            preprocessor = preprocessor.select_tilts(indices)
        selection = indices if self.selection is None else self.selection[indices]
        return _TiltSeriesImages(
            self.tilt_series.select_tilts(indices),
            preprocessor,
            cache=self.cache,
            selection=selection,
        )

    def _load(self, binsize: int) -> NDArray[np.float32]:
        tilt_series = self.tilt_series
        assert tilt_series.image_path is not None
        source = TiltImageSource(
            tilt_series.image_path, tilt_series.image_indices, tilt_series.ntilts
        )
        preprocessor = self.effective_preprocessor()

        def _process(t: int, image: NDArray[np.float32]) -> NDArray[np.float32]:
            if preprocessor is None:
                return fourier_bin(image, binsize)
            return preprocessor(image, index=t, scale=self.scale, binsize=binsize)

        # Each tilt image is read and preprocessed in the tasks of a dask graph, which
        # run in parallel threads. The number of the threads is limited to save memory.
        tasks = [
            dask.delayed(_process, pure=False)(
                t, dask.delayed(source.__getitem__, pure=False)(t)
            )
            for t in range(len(source))
        ]
        num_workers = _num_load_workers(source.image_shape, preprocessor)
        with _LOAD_LOCK:
            images = dask.compute(*tasks, scheduler="threads", num_workers=num_workers)
        out = np.stack(images, axis=0)
        # the images may be cached and shared by the loaders
        out.setflags(write=False)
        return out


def _num_load_workers(
    image_shape: tuple[int, int], preprocessor: TiltImagePreprocessor | None
) -> int:
    """Number of threads to preprocess tilt images within the memory budget."""
    # each thread has a tilt image, and the workspace to filter a patch of it
    image_bytes = image_shape[0] * image_shape[1] * 4
    if preprocessor is None:
        thread_bytes = 5 * image_bytes  # the image and its Fourier binning
    else:
        thread_bytes = image_bytes + preprocessor._workspace_nbytes(image_shape)
    by_memory = int(_LOAD_MEMORY_BUDGET // thread_bytes)
    return max(min(by_memory, os.cpu_count() or 1), 1)


def _as_tilt_series(tilt_series: TiltSeries) -> TiltSeries:
    if isinstance(tilt_series, TiltSeries):
        return tilt_series
    raise TypeError(f"Expected a TiltSeries, got {type(tilt_series)}.")


def _default_tilt_model(tilt_series: TiltSeries, tomo_hand: int) -> TiltSeriesModel:
    """Single-axis tilt model around the y axis from the tilt angles."""
    # mirroring along the z axis inverts the tilt angles
    angles = tilt_series.tilt_angles * tomo_hand
    return single_axis((float(angles.min()), float(angles.max())), axis="y")


def _same_tilt_images(ts0: TiltSeries, ts1: TiltSeries) -> bool:
    """Whether the tilt series have the same tilt images."""
    if ts0.image_path != ts1.image_path or ts0.scale != ts1.scale:
        return False
    if ts0.image_indices is None or ts1.image_indices is None:
        return ts0.image_indices is None and ts1.image_indices is None
    return np.array_equal(ts0.image_indices, ts1.image_indices)


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
        get_reconstructor: Callable[[], DirectFourierReconstructor],
        offsets: NDArray[np.float64],
        matrices: NDArray[np.float32],
        output_shape: tuple[int, int, int],
        order: int,
        batch_size: int,
    ):
        self.get_reconstructor = get_reconstructor  # loads the tilt images
        self.offsets = offsets  # (N, T, 2) positions from the center of the images
        self.matrices = matrices  # (N, 4, 4) from the subtomograms to the cubes
        self.output_shape = tuple(output_shape)
        self.order = order
        self.batch_size = batch_size

    def to_dask(self, backend: Backend, lazy: bool = True) -> da.Array:
        """Construct the dask array of the subtomograms.

        If ``lazy``, the reconstructor is created in the computation and released after
        that (see `_LazyReconstructor`). Otherwise, the reconstructor is created now
        and shared by the tasks.
        """
        output_shape = self.output_shape
        name = f"pseudo-subvolumes-{uuid.uuid4().hex}"
        nmole = self.offsets.shape[0]
        starts = range(0, nmole, self.batch_size)
        reconstructor: DirectFourierReconstructor | _LazyReconstructor
        if lazy:
            reconstructor = _LazyReconstructor(self.get_reconstructor, len(starts))
        else:
            reconstructor = self.get_reconstructor()
        zeros = (0,) * len(output_shape)
        task = _ReconstructionTask(output_shape, self.order, backend)
        layer = {}
        sizes = []
        for ib, start in enumerate(starts):
            sl = slice(start, start + self.batch_size)
            layer[(name, ib, *zeros)] = (
                task,
                reconstructor,
                ib,
                self.offsets[sl],
                self.matrices[sl],
            )
            sizes.append(self.offsets[sl].shape[0])
        graph = HighLevelGraph.from_collections(name, layer, dependencies=[])
        chunks = (tuple(sizes),) + tuple((s,) for s in output_shape)
        meta = np.empty((0,) * (len(output_shape) + 1), dtype=np.float32)
        return da.Array(graph, name, chunks, meta=meta)


class _LazyReconstructor:
    """The reconstructor shared by the reconstruction tasks of a dask array.

    The reconstructor is created (and the tilt images are loaded) by the first task,
    and released after all the tasks are computed, so the tilt images are in memory
    only during the computation, and loaded again if the array is computed again. The
    reconstruction tasks do not depend on a task that loads the tilt images, so that
    they are scheduled by their priority: the tilt images of many loaders are loaded
    one after another, rather than all at the beginning of the computation.
    """

    def __init__(self, factory: Callable[[], DirectFourierReconstructor], ntasks: int):
        self._factory = factory
        self._ntasks = ntasks
        self._lock = threading.Lock()
        self._reconstructor: DirectFourierReconstructor | None = None
        self._done: set[int] = set()

    def acquire(self) -> DirectFourierReconstructor:
        """Get the reconstructor, which is created if not yet."""
        with self._lock:
            if self._reconstructor is None:
                self._reconstructor = self._factory()
            return self._reconstructor

    def release(self, index: int) -> None:
        """Mark the task of given index as done."""
        with self._lock:
            self._done.add(index)
            # If only a part of the array is computed, the reconstructor is kept until
            # the rest are computed, or the array is deleted.
            if len(self._done) == self._ntasks:
                self._reconstructor = None
                self._done.clear()


class _ReconstructionTask:
    """Reconstruct a batch of subtomograms in a dask task."""

    def __init__(
        self,
        output_shape: tuple[int, int, int],
        order: int,
        backend: Backend,
    ):
        self.output_shape = output_shape
        self.order = order
        self.backend = backend

    def __call__(
        self,
        reconstructor: DirectFourierReconstructor | _LazyReconstructor,
        index: int,
        offsets: NDArray[np.float64],
        matrices: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        if not isinstance(reconstructor, _LazyReconstructor):
            return self._reconstruct(reconstructor, offsets, matrices)
        try:
            return self._reconstruct(reconstructor.acquire(), offsets, matrices)
        finally:
            reconstructor.release(index)

    def _reconstruct(
        self,
        reconstructor: DirectFourierReconstructor,
        offsets: NDArray[np.float64],
        matrices: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        xp = self.backend
        # (pixels from the center of the tilt images) -> (pixels of the tilt images)
        positions = offsets + np.array(reconstructor.images.shape[1:]) // 2
        cubes = reconstructor(positions)
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
