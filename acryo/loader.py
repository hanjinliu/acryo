# pyright: reportPrivateImportUsage=false

from __future__ import annotations
import psutil
from typing import (
    TYPE_CHECKING,
    Generic,
    Hashable,
    Iterator,
    NamedTuple,
    TypeVar,
    Any,
    overload,
)
import numpy as np
from dask import array as da
import polars as pl

from acryo._types import nm, pixel
from acryo._reader import imread
from acryo._loader_base import LoaderBase, Unset
from acryo.molecules import Molecules, MoleculeGroup
from acryo import _utils

if TYPE_CHECKING:
    from typing_extensions import Self
    from numpy.typing import NDArray
    from acryo.classification import PcaClassifier


_R = TypeVar("_R")
MEMORY_LIMIT = psutil.virtual_memory().total


class SubtomogramLoader(LoaderBase):
    """
    A class for efficient loading of subtomograms.

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
    """

    def __init__(
        self,
        image: np.ndarray | da.Array,
        molecules: Molecules,
        order: int = 3,
        scale: nm = 1.0,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
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

        super().__init__(
            order=order, scale=scale, output_shape=output_shape, corner_safe=corner_safe
        )

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
    ):
        dask_array, _scale = imread(path, chunks)
        if scale is None:
            scale = _scale
        return cls(
            dask_array,
            molecules=molecules,
            order=order,
            scale=scale,
            output_shape=output_shape,
            corner_safe=corner_safe,
        )

    @property
    def image(self) -> NDArray[np.float32] | da.Array:
        """Return tomogram image."""
        return self._image

    @property
    def molecules(self) -> Molecules:
        """Return the molecules of the subtomogram loader."""
        return self._molecules

    def __len__(self) -> int:
        """Return the number of subtomograms."""
        return self.molecules.pos.shape[0]

    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset | None = None,
        order: int | None = None,
        scale: float | None = None,
        corner_safe: bool | None = None,
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
        if corner_safe is None:
            corner_safe = self.corner_safe
        return self.__class__(
            self.image,
            molecules=molecules,
            output_shape=output_shape,
            order=order,
            scale=scale,
            corner_safe=corner_safe,
        )

    def binning(self, binsize: pixel = 2, *, compute: bool = True) -> Self:
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
        output_shape: pixel | tuple[pixel, ...] | None = None,
    ) -> list[da.Array]:
        """
        Construct a list of subtomogram lazy loader.

        Returns
        -------
        list of Delayed object
            Each object returns a subtomogram on execution by ``da.compute``.
        """
        if self._cache_available(output_shape):
            return [self._cached_dask_array[i] for i in range(len(self))]
        image = self.image
        scale = self.scale
        if isinstance(image, np.ndarray):
            image = da.from_array(image)

        output_shape = self._get_output_shape(output_shape)

        if self.corner_safe:
            _prep = _utils.prepare_affine_cornersafe
        else:
            _prep = _utils.prepare_affine
        tasks = []
        for i in range(len(self)):
            subvol, mtx = _prep(
                image,
                center=self.molecules.pos[i] / scale,
                output_shape=output_shape,
                rot=self.molecules.rotator[i],
            )
            task = _utils.rotated_crop(
                subvol,
                mtx,
                shape=output_shape,
                order=self.order,
                mode="constant",
                cval=np.mean,
            )
            tasks.append(da.from_delayed(task, shape=output_shape, dtype=np.float32))

        return tasks

    def construct_dask(
        self,
        output_shape: pixel | tuple[pixel, ...] | None = None,
    ) -> da.Array:
        """
        Construct a dask array of subtomograms.

        This function is always needed before parallel processing. If subtomograms
        are cached in a memory-map it will be used instead.

        Returns
        -------
        da.Array
            An 4-D array which ``arr[i]`` corresponds to the ``i``-th subtomogram.
        """
        if self._cache_available(output_shape):
            return self._cached_dask_array

        output_shape = self._get_output_shape(output_shape)
        tasks = self.construct_loading_tasks(output_shape=output_shape)
        out = da.stack(tasks, axis=0)
        return out

    def asnumpy(self, *, lim: int = MEMORY_LIMIT) -> NDArray[np.float32]:
        """Create a 4D image stack of all the subtomograms."""
        arr = self.construct_dask()
        if arr.nbytes > lim:
            raise MemoryError("The array is too large to be loaded into memory.")
        return arr.compute()

    @overload
    def groupby(self, by: str | pl.Expr) -> LoaderGroup[str]:
        ...

    @overload
    def groupby(self, by: list[str | pl.Expr]) -> LoaderGroup[tuple[str, ...]]:
        ...

    def groupby(self, by):
        mole_group = self.molecules.groupby(by)
        return LoaderGroup(
            mole_group,
            self.image,
            self.order,
            self.scale,
            self.output_shape,
            self.corner_safe,
        )

    def _get_output_shape(
        self, output_shape: pixel | tuple[pixel] | None
    ) -> tuple[pixel, ...]:
        if output_shape is None:
            if isinstance(self.output_shape, Unset):
                raise ValueError("Output shape is unknown.")
            _output_shape = self.output_shape
        else:
            _output_shape = _utils.normalize_shape(output_shape, ndim=self.image.ndim)
        return _output_shape


class ClassificationResult(NamedTuple):
    """Tuple of classification results."""

    loader: SubtomogramLoader
    classifier: PcaClassifier


_K = TypeVar("_K", bound=Hashable)


class LoaderGroup(Generic[_K]):
    """A groupby-like object for subtomogram loaders."""

    def __init__(
        self,
        group: MoleculeGroup,
        image,
        order: int,
        scale: float,
        output_shape: tuple[int, ...],
        corner_safe: bool,
    ):
        self._group = group
        self._image = image
        self._order = order
        self._scale = scale
        self._output_shape = output_shape
        self._corner_safe = corner_safe

    def __iter__(self) -> Iterator[tuple[_K, SubtomogramLoader]]:
        for key, mole in self._group:
            loader = SubtomogramLoader(
                self._image,
                mole,
                order=self._order,
                scale=self._scale,
                output_shape=self._output_shape,
                corner_safe=self._corner_safe,
            )
            yield key, loader

    def average(
        self, output_shape: tuple[int, ...] | None = None
    ) -> dict[_K, NDArray[np.float32]]:
        """Calculate average image."""
        if output_shape is None:
            output_shape = self._output_shape
        tasks = []
        keys: list[str] = []
        for key, loader in self:
            keys.append(key)
            dsk = loader.construct_dask(output_shape)
            tasks.append(da.mean(dsk, axis=0))

        out: list[NDArray[np.float32]] = da.compute(tasks)[0]
        return {key: img for key, img in zip(keys, out)}


def get_feature_list(corr_max, local_shifts, rotvec) -> list[pl.Series]:
    features = [
        pl.Series("score", corr_max),
        pl.Series("shift-z", np.round(local_shifts[:, 0], 2)),
        pl.Series("shift-y", np.round(local_shifts[:, 1], 2)),
        pl.Series("shift-x", np.round(local_shifts[:, 2], 2)),
        pl.Series("rotvec-z", np.round(rotvec[:, 0], 5)),
        pl.Series("rotvec-y", np.round(rotvec[:, 1], 5)),
        pl.Series("rotvec-x", np.round(rotvec[:, 2], 5)),
    ]
    return features


def check_input(
    order: int,
    output_shape: pixel | tuple[pixel, pixel, pixel] | Unset,
    scale: float,
    corner_safe: bool,
    ndim: int,
):
    # check interpolation order
    if order not in (0, 1, 3):
        raise ValueError(
            f"The third argument 'order' must be 0, 1 or 3, got {order!r}."
        )

    # check output_shape
    if isinstance(output_shape, Unset):
        _output_shape = output_shape
    else:
        _output_shape = _utils.normalize_shape(output_shape, ndim=ndim)

    # check scale
    _scale = float(scale)
    if _scale <= 0:
        raise ValueError("Negative scale is not allowed.")

    _corner_safe = bool(corner_safe)
    return order, _output_shape, _scale, _corner_safe
