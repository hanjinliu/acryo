# pyright: reportPrivateImportUsage=false

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
import polars as pl
from dask import array as da
from dask.array.core import normalize_chunks
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph

from acryo._types import nm, pixel
from acryo._reader import imread
from acryo.molecules import Molecules
from acryo.backend import Backend
from acryo import _utils
from acryo.loader._base import LoaderBase, Unset, _ShapeType, INDEX_OPT
from acryo.loader._extracted import ExtractedSubvolumeLoader, SUBVOLUME_INDEX
from acryo.tilt import TiltSeriesModel, NoWedge
from acryo._dask import DaskArrayList, subvolume_array_list, subvolume_stack

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

        To process many subtomograms at once, ``construct_dask`` is more efficient.

        Returns
        -------
        DaskArrayList
            Each dask array returns a subtomogram on computation.
        """
        output_shape = self._get_output_shape(output_shape)
        return subvolume_array_list(
            *self._prep_subvolumes(output_shape),
            output_shape=output_shape,
            order=self.order,
            backend=backend or Backend(),
        )

    def construct_dask(
        self,
        output_shape: pixel | tuple[pixel, ...] | None = None,
        backend: Backend | None = None,
    ) -> da.Array:
        """Construct a dask array of subtomograms.

        This function is always needed before parallel processing. Each subtomogram
        is a chunk of the returned array. The tomogram chunks are shared, that is,
        each tomogram chunk is loaded only once and all the subtomograms in it are
        cropped from it.

        Returns
        -------
        da.Array
            An 4-D array which ``arr[i]`` corresponds to the ``i``-th subtomogram.
        """
        output_shape = self._get_output_shape(output_shape)
        return subvolume_stack(
            *self._prep_subvolumes(output_shape),
            output_shape=output_shape,
            order=self.order,
            backend=backend or Backend(),
        )

    def _prep_subvolumes(
        self, output_shape: tuple[pixel, pixel, pixel]
    ) -> tuple[da.Array, NDArray[np.intp], NDArray[np.intp], NDArray[np.float32]]:
        image = self.image
        if isinstance(image, np.ndarray):
            # a single chunk is enough because slicing a numpy array is cheap
            image = da.from_array(image, chunks=image.shape, name=False)
        mole = self.molecules
        if mole.count() == 0:
            empty = np.zeros((0, 3), dtype=np.intp)
            return image, empty, empty, np.zeros((0, 4, 4), dtype=np.float32)
        try:
            starts, stops, matrices = _utils.prepare_affine_regions(
                mole.pos / self.scale,
                mole.rotator,
                img_shape=image.shape,
                output_shape=output_shape,
                order=self.order,
                corner_safe=self.corner_safe,
            )
        except _utils.SubvolumeOutOfBoundError as err:
            i = err.index
            raise err.with_msg(
                f"The {i}-th molecule at {tuple(mole.pos[i])} is out of bound. "
                f"{err.msg}"
            )
        return image, starts, stops, matrices

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
    """Class for saving extracted subtomograms."""

    def __init__(self, save_dir):
        self._save_dir = Path(save_dir)
        self._save_dir.mkdir()

    def path_i(self, i: int) -> Path:
        return self._save_dir / f"{i:08d}.npy"

    def save(self, array: NDArray[np.float32], i: int) -> None:
        np.save(self.path_i(i), array)

    def to_dask(self, num: int, shape, chunksize: int = 50) -> da.Array:
        if num == 0:
            return da.zeros((0, *shape), dtype=np.float32)
        chunks = normalize_chunks((max(chunksize, 1), *shape), (num, *shape))
        name = "load-subvolumes-" + tokenize(str(self._save_dir), num, chunks)
        zeros = (0,) * len(shape)
        layer = {}
        start = 0
        for ib, size in enumerate(chunks[0]):
            layer[(name, ib, *zeros)] = (self.load_arrays, start, start + size)
            start += size
        graph = HighLevelGraph.from_collections(name, layer, dependencies=[])
        return da.Array(graph, name, chunks, dtype=np.float32)

    def load_arrays(self, start: int, stop: int) -> NDArray[np.float32]:
        return np.stack([np.load(self.path_i(i)) for i in range(start, stop)], axis=0)
