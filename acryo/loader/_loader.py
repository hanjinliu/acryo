# pyright: reportPrivateImportUsage=false

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
import polars as pl
from dask import array as da, delayed

from acryo._types import nm, pixel
from acryo._reader import imread
from acryo.molecules import Molecules
from acryo.backend import Backend
from acryo import _utils
from acryo.loader._base import LoaderBase, Unset, _ShapeType, INDEX_OPT
from acryo.loader._extracted import ExtractedSubvolumeLoader, SUBVOLUME_INDEX
from acryo.tilt import TiltSeriesModel, NoWedge
from acryo._dask import DaskTaskPool, DaskArrayList

if TYPE_CHECKING:
    from typing_extensions import Self
    from numpy.typing import NDArray


class SubtomogramLoader(LoaderBase):
    """A class for efficient loading of subtomograms.

    A ``SubtomogramLoader`` instance is basically composed of two elements,
    an image and a Molecules object. A subtomogram is loaded by creating a
    local rotated Cartesian coordinate at a molecule and calculating mapping
    from the image to the subtomogram.

    Parameters
    ----------
    image : np.ndarray or da.Array
        Tomogram image. Must be 3-D.
    molecules : Molecules
        Molecules object that represents positions and orientations of
        subtomograms.
    order : int, default is 3
        Interpolation order of subtomogram sampling.
        - 0 = Nearest neighbor
        - 1 = Linear interpolation
        - 3 = Cubic interpolation
    scale : float, default is 1.0
        Physical scale of pixel, such as nm. This value does not affect
        averaging/alignment results but molecule coordinates are multiplied
        by this value. This parameter is useful when another loader with
        binned image is created.
    output_shape : int or tuple of int, optional
        Shape of output subtomogram in pixel. This parameter is not required
        if template (or mask) image is available immediately.
    corner_safe : bool, default is False
        If true, regions around molecules will be cropped at a volume larger
        than ``output_shape`` so that densities at the corners will not be
        lost due to rotation. If target density is globular, this parameter
        should be set false to save computation time.
    tilt_model : TiltSeriesModel, optional
        Tilt series model to be used for alignment.
    """

    def __init__(
        self,
        image: np.ndarray | da.Array,
        molecules: Molecules,
        order: int = 3,
        scale: nm = 1.0,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
        tilt_model: TiltSeriesModel | None = None,
    ) -> None:
        # check type of input image
        if not isinstance(image, (np.ndarray, da.Array)):
            raise TypeError(
                "Input image of a SubtomogramLoader instance must be np.ndarray "
                f"or dask.Array, got {type(image)}."
            )

        self._image = image

        # check type of molecules
        if not isinstance(molecules, Molecules):
            raise TypeError(
                "The second argument 'molecules' must be a Molecules object, got"
                f"{type(molecules)}."
            )
        self._molecules = molecules
        self._tilt_model = tilt_model or NoWedge()
        self._corner_safe = corner_safe
        super().__init__(order=order, scale=scale, output_shape=output_shape)

    def __repr__(self) -> str:
        shape = self.image.shape
        mole_repr = repr(self.molecules)
        return (
            f"{self.__class__.__name__}(tomogram={shape}, molecules={mole_repr}, "
            f"output_shape={self.output_shape}, order={self.order}, "
            f"scale={self.scale:.4f})"
        )

    @classmethod
    def imread(
        cls,
        path: str,
        molecules: Molecules,
        order: int = 3,
        scale: nm | None = None,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
        chunks: Any = "auto",
        tilt_model: TiltSeriesModel | None = None,
    ):
        dask_array, _scale = imread(str(path), chunks)
        if scale is None:
            scale = _scale
        return cls(
            dask_array,
            molecules=molecules,
            order=order,
            scale=scale,
            output_shape=output_shape,
            corner_safe=corner_safe,
            tilt_model=tilt_model,
        )

    @property
    def image(self) -> NDArray[np.float32] | da.Array:
        """Return tomogram image."""
        return self._image

    @property
    def molecules(self) -> Molecules:
        """Return the molecules of the subtomogram loader."""
        return self._molecules

    @property
    def corner_safe(self) -> bool:
        """Return whether the loader is corner-safe."""
        return self._corner_safe

    @property
    def tilt_model(self) -> TiltSeriesModel:
        """Return the tilt model of the subtomogram loader."""
        return self._tilt_model

    def __len__(self) -> int:
        """Return the number of subtomograms."""
        return self.molecules.pos.shape[0]

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
        return self.__class__(
            self.image,
            molecules=molecules,
            output_shape=output_shape,
            order=order,
            scale=scale,
            tilt_model=self.tilt_model,
        )

    def order_optimize(self) -> Self:
        """Sort molecules by their positions.

        When the tomogram is chunked, nearby molecules should be loaded in the same
        timing, otherwise the disk access will be extremely slow. This method sorts
        the molecules by their (z, y, x) positions to optimize the loading speed.
        To restore the original order, use `order_restore` method later.
        """
        img = self.image
        if isinstance(img, np.ndarray):
            return self  # no need to optimize for numpy array
        zchunks, ychunks, xchunks = img.chunks
        zcum = np.cumsum([0] + list(zchunks))
        ycum = np.cumsum([0] + list(ychunks))
        xcum = np.cumsum([0] + list(xchunks))
        df = (
            self.molecules.to_dataframe()
            .with_columns(pl.arange(pl.len()).alias(INDEX_OPT))
            .sort(
                pl.col("y").truediv(self.scale).cut(ycum),
                pl.col("x").truediv(self.scale).cut(xcum),
                pl.col("z").truediv(self.scale).cut(zcum),
            )
        )
        mole = Molecules.from_dataframe(df)
        return self.replace(molecules=mole)

    def binning(self, binsize: pixel = 2, *, compute: bool = True) -> Self:
        """Return a new instance with binned image.

        This method also properly translates the molecule coordinates.

        Parameters
        ----------
        binsize : int, default is 2
            Bin size.
        compute : bool, default is True
            If true, the image is computed immediately to a numpy array.

        Returns
        -------
        SubtomogramLoader
            A new instance with binned image.
        """
        if binsize == 1:
            return self.copy()
        tr = -(binsize - 1) / 2 * self.scale
        molecules = self.molecules.translate([tr, tr, tr])
        binned_image = _utils.bin_image(self.image, binsize=binsize)
        if isinstance(binned_image, da.Array) and compute:
            binned_image = binned_image.compute()
        out = self.replace(
            molecules=molecules,
            scale=self.scale * binsize,
        )

        out._image = binned_image
        return out

    def construct_loading_tasks(
        self,
        output_shape: _ShapeType = None,
        backend: Backend | None = None,
    ) -> DaskArrayList:
        """Construct a list of subtomogram lazy loader.

        Returns
        -------
        list of Delayed object
            Each object returns a subtomogram on execution by ``da.compute``.
        """
        output_shape = self._get_output_shape(output_shape)
        xp = backend or Backend()

        image = self.image
        scale = self.scale
        if isinstance(image, np.ndarray):
            image = da.from_array(image, asarray=xp.asarray)

        if self.corner_safe:
            _prep = _utils.prepare_affine_cornersafe
        else:
            _prep = _utils.prepare_affine
        pool = DaskTaskPool.from_func(xp.rotated_crop)
        for i in range(self.molecules.count()):
            try:
                subvol, mtx = _prep(
                    image,
                    center=self.molecules.pos[i] / scale,
                    output_shape=output_shape,
                    rot=self.molecules.rotator[i],
                    order=self.order,
                )
            except _utils.SubvolumeOutOfBoundError as err:
                raise err.with_msg(
                    f"The {i}-th molecule at {tuple(self.molecules.pos[i])} is "
                    f"out of bound. {err.msg}"
                )
            pool.add_task(
                subvol,
                mtx,
                shape=output_shape,
                order=self.order,
                cval=xp.mean,
            )

        return pool.asarrays(shape=output_shape, dtype=np.float32)

    def extract_subtomograms(
        self, save_path, chunksize: int = 1
    ) -> ExtractedSubvolumeLoader:
        store_target = ArrayStoreInterface(save_path)
        num_mole = self.molecules.count()
        tasks = self.construct_mapping_tasks(
            store_target.save,
            output_shape=self.output_shape,
            var_kwarg={"i": np.arange(num_mole)},
        )
        tasks.compute()
        dsk_round = store_target.to_dask(
            num=num_mole,
            shape=self.output_shape,
            chunksize=chunksize,
        )
        index = pl.arange(num_mole).alias(SUBVOLUME_INDEX)
        return ExtractedSubvolumeLoader(
            dsk_round,
            molecules=self.molecules.with_features(index),
            order=self.order,
            scale=self.scale,
            tilt_model=self.tilt_model,
        )

    def _default_align_kwargs(self) -> dict[str, Any]:
        """Return default keyword arguments for alignment."""
        return {"tilt": self.tilt_model}


class ArrayStoreInterface:
    def __init__(self, save_dir):
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir()

    def path_i(self, i: int) -> Path:
        return self._save_dir / f"{i:08d}.npy"

    def save(self, array: NDArray[np.float32], i: int) -> None:
        np.save(self.path_i(i), array)

    def to_dask(self, num: int, shape, chunksize: int = 50) -> da.Array:
        arrays = [
            da.from_delayed(
                self.load_array(i),
                shape=shape,
                dtype=np.float32,
                name=f"load-{id(self)}-{i}",
            )
            for i in range(num)
        ]
        out = da.stack(arrays, axis=0)
        if chunksize > 1:
            out = out.rechunk((chunksize, *shape))
        return out

    @delayed
    def load_array(self, i: int) -> NDArray[np.float32]:
        return np.load(self.path_i(i))
