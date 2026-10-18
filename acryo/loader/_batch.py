# pyright: reportPrivateImportUsage=false

from __future__ import annotations
from types import MappingProxyType

import math
import weakref
from typing import Hashable, Iterable, Iterator, Mapping, TYPE_CHECKING
import numpy as np
import polars as pl

from dask import array as da
from scipy.spatial.transform import Rotation

from acryo import _utils
from acryo.alignment._base import TomographyInput
from acryo.molecules import Molecules
from acryo.backend import Backend
from acryo._types import nm, pixel
from acryo._dask import DaskArrayList, DaskTaskList
from acryo._polars_compat import bin_index
from acryo.loader import _misc
from acryo.loader._base import LoaderBase, MaskInputType, TemplateInputType, INDEX_OPT
from acryo.loader._loader import SubtomogramLoader, Unset

from acryo.alignment import BaseAlignmentModel, ZNCCAlignment, AlignmentFactory
from acryo.tilt._base import TiltSeriesModel

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self
    from acryo.loader._base import _ShapeType

IMAGE_ID_LABEL = "image-id"


class BatchLoader(LoaderBase):
    """Collection of tomograms and their molecules.

    A `BatchLoader` is similar to a list of `SubtomogramLoader` objects, but with
    better consistency in loader parameters and more convenient access to the molecules.
    This class is useful for processing many tomograms at once, especially when you want
    to get the average of all the molecules available.

    Tomograms are stored as images, and loaded by `SubtomogramLoader`. Other loaders,
    such as `BackProjectionLoader`, can also be added by `add_loader`. They are stored
    as is, and the subtomograms are loaded by them with the molecules, order, scale and
    output shape of the batch loader.
    """

    def __init__(
        self,
        order: int = 3,
        scale: nm = 1.0,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
    ) -> None:
        super().__init__(order, scale, output_shape)
        self._images: dict[Hashable, NDArray[np.float32] | da.Array] = {}
        self._molecules: Molecules = Molecules.empty([IMAGE_ID_LABEL])
        self._tilt_models: dict[Hashable, TomographyInput] = {}
        # loaders other than `SubtomogramLoader`. Their molecules are not used.
        self._sources: dict[Hashable, LoaderBase] = {}

    def __repr__(self) -> str:
        loaders_repr = []
        for loader in self.loaders:
            mole_repr = repr(loader.molecules)
            if isinstance(loader, SubtomogramLoader):
                shape = loader.image.shape
                loaders_repr.append(
                    f"{loader.__class__.__name__}(tomogram={shape}, "
                    f"molecules={mole_repr})"
                )
            else:
                loaders_repr.append(
                    f"{loader.__class__.__name__}(molecules={mole_repr})"
                )
        loaders_repr = ", ".join(loaders_repr)
        return (
            f"{self.__class__.__name__}(loaders=[{loaders_repr}], "
            f"output_shape={self.output_shape}, order={self.order}, "
            f"scale={self.scale:.4f})"
        )

    @property
    def loaders(self) -> LoaderAccessor:
        """Interface to access the subtomogram loaders.

        Examples
        --------
        1. Get the loader for image ID 1

            ```python
            batch_loader.loaders[1]
            ```

        2. Iterate over all loaders

            ```python
            for loader in batch_loader.loaders:
                ...  # do something with loader
            ```

        3. Iterate over all loaders with image IDs

            ```python
            for img_id, loader in batch_loader.iter_loaders():
                ...  # do something with loader
            ```
        """
        return LoaderAccessor(self)

    @property
    def images(self) -> MappingProxyType:
        """All the images in the collection.

        Loaders added by `add_loader` other than `SubtomogramLoader` are not included.
        """
        return MappingProxyType(self._images)

    @property
    def tilt_models(self) -> MappingProxyType[Hashable, TomographyInput]:
        """All the tilt series models in the collection."""
        tilt_models = dict(self._tilt_models)
        for image_id, source in self._sources.items():
            tilt_models[image_id] = source._default_align_kwargs().get("tilt")
        return MappingProxyType(tilt_models)

    def add_tomogram(
        self,
        image: np.ndarray | da.Array,
        molecules: Molecules,
        image_id: Hashable | None = None,
        tilt_model: TiltSeriesModel | None = None,
    ) -> Self:
        """Add a tomogram and its molecules to the collection.

        Parameters
        ----------
        image : np.ndarray or da.Array
            Tomogram image. This argument is passed directly to the `SubtomogramLoader`
            constructor.
        molecules : Molecules
            Molecules in the tomogram (corresponding to the ``image`` argument). This
            argument is passed directly to the `SubtomogramLoader` constructor.
        image_id : Hashable, optional
            Identifier for the tomogram. If not provided, a unique identifier will be
            generated. This identifier is used to tag molecules with the tomogram they
            reside in.
        """
        new = self._register_tomogram(image, molecules, image_id, tilt_model)
        self._molecules = _concat_molecules([self._molecules, new])
        return self

    def add_loader(self, loader: LoaderBase, image_id: Hashable | None = None) -> Self:
        """Add a loader to the collection.

        A `SubtomogramLoader` is added as a tomogram (see `add_tomogram`), and the
        loaders in a `BatchLoader` are added one by one. Other loaders, such as
        `BackProjectionLoader`, are added as is. The subtomograms are loaded by the
        loader with the molecules, order and scale of the batch loader, and with its
        output shape if set.

        Parameters
        ----------
        loader : LoaderBase
            Loader to be added.
        image_id : Hashable, optional
            Identifier for the tomogram of the loader. If not provided, a unique
            identifier will be generated. Not allowed for a `BatchLoader`.
        """
        new = self._register_loader(loader, image_id)
        self._molecules = _concat_molecules([self._molecules, *new])
        return self

    def _register_tomogram(
        self,
        image: np.ndarray | da.Array,
        molecules: Molecules,
        image_id: Hashable | None,
        tilt_model: TiltSeriesModel | None,
    ) -> Molecules:
        """Register a tomogram and return its molecules tagged with the image ID."""
        if image_id is None:
            image_id = self._new_image_id()
        elif image_id in self._sources:
            raise ValueError(f"Image ID {image_id!r} is already used by a loader.")
        tagged = _tag_molecules(molecules, image_id)
        self._images[image_id] = image
        self._tilt_models[image_id] = tilt_model
        return tagged

    def _register_loader(
        self, loader: LoaderBase, image_id: Hashable | None
    ) -> list[Molecules]:
        """Register a loader and return its molecules tagged with the image IDs."""
        if isinstance(loader, SubtomogramLoader):
            tagged = self._register_tomogram(
                loader.image, loader.molecules, image_id, loader.tilt_model
            )
            return [tagged]
        elif isinstance(loader, BatchLoader):
            if image_id is not None:
                raise TypeError(
                    f"Cannot specify 'image_id' for {type(loader).__name__}."
                )
            return [
                m for sub in loader.loaders for m in self._register_loader(sub, None)
            ]
        elif not isinstance(loader, LoaderBase):
            raise TypeError(f"Cannot add {type(loader)}.")
        if image_id is None:
            image_id = self._new_image_id()
        elif image_id in self._images:
            raise ValueError(f"Image ID {image_id!r} is already used by a tomogram.")
        tagged = _tag_molecules(loader.molecules, image_id)
        self._sources[image_id] = loader
        return [tagged]

    def _new_image_id(self) -> int:
        image_id = len(self._images) + len(self._sources)
        while image_id in self._images or image_id in self._sources:
            image_id += 1
        return image_id

    def _image_rows(self) -> list[tuple[Hashable, slice | NDArray[np.intp]]]:
        """Image IDs and the rows of their molecules, in the order of `self.loaders`.

        The images are in the order of the first appearance, and the rows are given as
        a slice if they are contiguous.
        """
        image_ids = self.features[IMAGE_ID_LABEL].to_numpy()
        if image_ids.size == 0:
            return []
        keys, first, inverse = np.unique(
            image_ids, return_index=True, return_inverse=True
        )
        group_order = np.argsort(first)
        rank = np.empty_like(group_order)
        rank[group_order] = np.arange(group_order.size)
        group_of_rows = rank[inverse.ravel()]
        rows = np.argsort(group_of_rows, kind="stable")
        stops = np.cumsum(np.bincount(group_of_rows))
        out: list[tuple[Hashable, slice | NDArray[np.intp]]] = []
        start = 0
        for igroup, stop in zip(group_order, stops):
            sub = rows[start:stop]
            if sub[-1] - sub[0] + 1 == sub.size:
                spec: slice | NDArray[np.intp] = slice(int(sub[0]), int(sub[-1]) + 1)
            else:
                spec = sub
            key = keys[igroup]
            out.append((key.item() if isinstance(key, np.generic) else key, spec))
            start = stop
        return out

    def _iter_image_molecules(
        self,
    ) -> Iterator[tuple[Hashable, slice | NDArray[np.intp], Molecules]]:
        """Iterate over the image IDs, the rows and the molecules of each image.

        Same as ``self.molecules.subset(rows)`` for each image, but the rotations are
        converted only once.
        """
        mole = self.molecules
        pos = mole.pos
        quat = mole.quaternion(canonical=False)
        features = mole.features
        for key, rows in self._image_rows():
            yield key, rows, Molecules(pos[rows], Rotation(quat[rows]), features[rows])

    def _make_loader(self, image_id: Hashable, molecules: Molecules) -> LoaderBase:
        """Create the loader of the given image ID with the given molecules."""
        if image_id in self._images:
            return SubtomogramLoader(
                self._images[image_id],
                molecules,
                order=self.order,
                scale=self.scale,
                output_shape=self.output_shape,
                tilt_model=self._tilt_models.get(image_id, None),
            )
        source = self._sources[image_id]
        # keep the output shape of the loader if not set, as some loaders cannot change
        # it. The scale is not replaced if not changed, because some loaders, such as
        # `BackProjectionLoader`, convert it to other parameters.
        return source.replace(
            molecules=molecules,
            output_shape=self.output_shape or None,
            order=self.order,
            scale=None if math.isclose(source.scale, self.scale) else self.scale,
        )

    @classmethod
    def from_loaders(
        cls,
        loaders: Iterable[LoaderBase] | Mapping[Hashable, LoaderBase],
        order: int = 3,
        scale: nm = 1.0,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
    ) -> Self:
        """Construct a loader from a list of loaders.

        If a mapping is given, its keys are used as the image IDs of the loaders (see
        `add_loader`). Otherwise, the image IDs are generated, so the image IDs of a
        `BatchLoader` split into multiple loaders are not kept.
        """
        self = cls(
            order=order,
            scale=scale,
            output_shape=output_shape,
        )
        if isinstance(loaders, Mapping):
            items = list(loaders.items())
        else:
            items = [(None, ldr) for ldr in loaders]
        # the molecules are concatenated at once, which is much faster for many loaders
        new = [mole for _id, ldr in items for mole in self._register_loader(ldr, _id)]
        self._molecules = _concat_molecules([self._molecules, *new])
        return self

    @property
    def molecules(self) -> Molecules:
        """Collect all the molecules from all the loaders"""
        return self._molecules

    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: _ShapeType | Unset = None,
        order: int | None = None,
        scale: float | None = None,
    ) -> Self:
        """Return a new instance with different parameter(s)."""
        if output_shape is None:
            output_shape = self.output_shape
        if order is None:
            order = self.order
        if scale is None:
            scale = self.scale
        out = type(self)(
            order=order,
            scale=scale,
            output_shape=output_shape,
        )
        out._images = self._images.copy()
        out._tilt_models = self._tilt_models.copy()
        out._sources = self._sources.copy()
        if molecules is None:
            out._molecules = self._molecules.copy()
        else:
            out._molecules = molecules
            _id_exists = set(molecules.features[IMAGE_ID_LABEL].unique())
            for k in list(out._images.keys()):
                if k not in _id_exists:
                    out._images.pop(k)
            for k in list(out._sources.keys()):
                if k not in _id_exists:
                    out._sources.pop(k)
        return out

    def order_optimize(self) -> Self:
        """Sort molecules by their positions.

        When the tomogram is chunked, nearby molecules should be loaded in the same
        timing, otherwise the disk access will be extremely slow. This method sorts
        the molecules by their (z, y, x) positions to optimize the loading speed.
        To restore the original order, use `order_restore` method later.
        """
        df = self.molecules.to_dataframe().with_columns(
            pl.arange(pl.len()).alias(INDEX_OPT)
        )
        df_sorted = []
        for sl, sub in df.group_by(IMAGE_ID_LABEL):
            # sl is something like (1,)
            img = self._images.get(sl[0])
            if not isinstance(img, da.Array):
                # numpy arrays and other loaders are not chunked
                df_sorted.append(sub)
            else:
                zchunks, ychunks, xchunks = img.chunks
                zcum = np.cumsum([0] + list(zchunks))
                ycum = np.cumsum([0] + list(ychunks))
                xcum = np.cumsum([0] + list(xchunks))
                df_sub = sub.sort(
                    bin_index(pl.col("y").truediv(self.scale), ycum),
                    bin_index(pl.col("x").truediv(self.scale), xcum),
                    bin_index(pl.col("z").truediv(self.scale), zcum),
                )
                df_sorted.append(df_sub)
        df = pl.concat(df_sorted)
        mole = Molecules.from_dataframe(df)
        return self.replace(molecules=mole)

    def binning(self, binsize: int = 2, *, compute: bool = False) -> Self:
        """Return a new instance with binned images.

        Loaders other than `SubtomogramLoader` are binned by their `binning` method.
        The output shape in pixels is kept, so if the output shape of a loader is
        binned together, such as `ExtractedSubvolumeLoader`, the output shape has to
        be replaced by `replace(output_shape=...)`.
        """
        if binsize == 1:
            return self.copy()
        tr = -(binsize - 1) / 2 * self.scale
        shifts = np.full((self.molecules.count(), 3), tr)
        _sources: dict[Hashable, LoaderBase] = {}
        for _id, rows, mole in self._iter_image_molecules():
            if _id not in self._sources:
                continue
            # the molecules may be translated by binning, depending on the loader
            loader = self._make_loader(_id, mole)
            binned = loader.binning(binsize, compute=compute)
            shifts[rows] = binned.molecules.pos - loader.molecules.pos
            _sources[_id] = binned
        for _id, source in self._sources.items():
            if _id not in _sources:  # no molecules
                _sources[_id] = source.binning(binsize, compute=compute)
        molecules = self.molecules.translate(shifts)
        _images: dict[Hashable, NDArray[np.float32] | da.Array] = {}
        _compute_dict: dict[Hashable, da.Array] = {}
        for _id, image in self._images.items():
            binned_image = _utils.bin_image(image, binsize=binsize)
            if isinstance(binned_image, da.Array) and compute:
                _compute_dict[_id] = binned_image
            _images[_id] = binned_image

        if _compute_dict:
            (_computed,) = da.compute(_compute_dict)
            _images.update(_computed)

        out = self.replace(
            molecules=molecules,
            scale=self.scale * binsize,
        )
        out._images = _images
        out._sources = _sources
        return out

    def _molecule_order(self) -> NDArray[np.intp] | None:
        """Indices to sort the outputs of `self.loaders` to the order of the molecules.

        The loaders are in the order of the first appearance of the image IDs, and each
        of them has the molecules of the image in the original order. None is returned
        if the outputs are already in the order of the molecules, that is, the
        molecules of each image are contiguous.
        """
        image_rows = self._image_rows()
        if all(isinstance(rows, slice) for _, rows in image_rows):
            return None
        indices = np.arange(self.molecules.count())
        order = np.concatenate([indices[rows] for _, rows in image_rows])
        return np.argsort(order)

    def construct_loading_tasks(
        self,
        output_shape: _ShapeType = None,
        backend: Backend | None = None,
    ) -> DaskArrayList:
        """Construct batch loading tasks."""
        _backend = backend or Backend()
        tasks = DaskArrayList.concat(
            loader.construct_loading_tasks(output_shape=output_shape, backend=_backend)
            for loader in self.loaders
        )
        if (indices := self._molecule_order()) is not None:
            tasks = DaskArrayList(tasks[i] for i in indices)
        return tasks

    def construct_dask(
        self,
        output_shape: _ShapeType = None,
        backend: Backend | None = None,
    ) -> da.Array:
        """Construct a dask array of subtomograms of all the tomograms."""
        _backend = backend or Backend()
        out = da.concatenate(
            [
                loader.construct_dask(output_shape=output_shape, backend=_backend)
                for loader in self.loaders
            ],
            axis=0,
        )
        if (indices := self._molecule_order()) is not None:
            out = out[indices]
        return out

    def construct_wedge_weighted_ft(
        self,
        output_shape: tuple[pixel, pixel, pixel],
        backend: Backend,
    ) -> da.Array:
        # each tomogram has its own tilt series model
        out = da.concatenate(
            [
                loader.construct_wedge_weighted_ft(output_shape, backend)
                for loader in self.loaders
            ],
            axis=0,
        )
        if (indices := self._molecule_order()) is not None:
            out = out[indices]
        return out

    def align(
        self,
        template: TemplateInputType,
        *,
        mask: MaskInputType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] | AlignmentFactory = ZNCCAlignment,
        backend: Backend | None = None,
        **align_kwargs,
    ) -> Self:
        """Align subtomograms to the template image.

        This method conduct so called "subtomogram alignment". Only shifts and rotations
        are calculated in this method. To get averaged image, you'll have to run
        "average" method using the resulting SubtomogramLoader instance.

        Parameters
        ----------
        template : 3D array or ImageProvider
            Template image. If ImageProvider is given, the image will be provided
            accordingly using the scale of the loader object.
        mask : np.ndarray or callable of np.ndarray to np.ndarray optional
            Mask image. Must in the same shape as the template.
        max_shifts : int or tuple of int, default is (1., 1., 1.)
            Maximum shift between subtomograms and template.
        alignment_model : subclass of BaseAlignmentModel, optional
            Alignment model class used for subtomogram alignment. By default,
            ``ZNCCAlignment`` will be used.
        align_kwargs : optional keyword arguments
            Additional keyword arguments passed to the input alignment model.

        Returns
        -------
        subtomogram loader object
            A loader instance of the same type with updated molecules.
        """
        _backend = backend or Backend()
        max_shifts = _misc.normalize_max_shifts(max_shifts)
        tasks = DaskTaskList([])
        is_multi = False
        for each in self.loaders:
            model = alignment_model(
                self.normalize_template(template, allow_multiple=True),
                self.normalize_mask(mask),
                **each._update_align_kwargs(align_kwargs),
            )
            if model.has_hetero_templates:
                is_multi = True
                tasks.extend(
                    each._prep_align_multi_templates(model, max_shifts, _backend)
                )
            else:
                tasks.extend(each._prep_align_tasks(model, max_shifts, _backend))

        all_results = tasks.compute()
        if (indices := self._molecule_order()) is not None:
            all_results = [all_results[i] for i in indices]

        if is_multi:
            return self._post_align_multi_templates(all_results, model.input_shape)
        return self._post_align(all_results, model.input_shape)

    def construct_landscape(
        self,
        template: TemplateInputType,
        *,
        mask: MaskInputType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] | AlignmentFactory = ZNCCAlignment,
        upsample: int = 1,
        **align_kwargs,
    ) -> da.Array:
        arrays: list[da.Array] = []
        for each in self.loaders:
            arr = each.construct_landscape(
                template,
                mask=mask,
                max_shifts=max_shifts,
                alignment_model=alignment_model,
                upsample=upsample,
                **align_kwargs,
            )
            arrays.append(arr)
        out = da.concatenate(arrays, axis=0)
        if (indices := self._molecule_order()) is not None:
            out = out[indices]
        return out

    def score(
        self,
        templates: list[TemplateInputType],
        *,
        mask: MaskInputType = None,
        alignment_model: type[BaseAlignmentModel] | AlignmentFactory = ZNCCAlignment,
        **align_kwargs,
    ) -> list[NDArray[np.float32]]:
        scores = []
        for each in self.loaders:
            scores.append(
                each.score(
                    templates,
                    mask=mask,
                    alignment_model=alignment_model,
                    **align_kwargs,
                )
            )
        out = np.concatenate(scores, axis=1)
        if (indices := self._molecule_order()) is not None:
            out = out[:, indices]
        return out


def _tag_molecules(molecules: Molecules, image_id: Hashable) -> Molecules:
    """Return a copy of the molecules tagged with the image ID."""
    molecules = molecules.copy()
    molecules.features = molecules.features.with_columns(
        pl.Series(IMAGE_ID_LABEL, np.full((len(molecules)), image_id))
    )
    return molecules


def _concat_molecules(molecules: list[Molecules]) -> Molecules:
    """Concatenate the molecules, ignoring the empty ones as `Molecules.concat_with`."""
    nonempty = [mole for mole in molecules if mole.count() > 0]
    if len(nonempty) == 0:
        return molecules[0]
    elif len(nonempty) == 1:
        return nonempty[0]
    return Molecules.concat(nonempty)


class LoaderAccessor:
    """The interface to access subtomogram loaders in a BatchLoader."""

    def __init__(self, collection: BatchLoader):
        self._loader = weakref.ref(collection)

    def __getitem__(self, idx: Hashable) -> LoaderBase:
        ldr = self._get_loader()
        rows = dict(ldr._image_rows()).get(idx)
        if rows is None:
            raise KeyError(idx)
        return ldr._make_loader(idx, ldr.molecules.subset(rows))

    def __iter__(self) -> Iterator[LoaderBase]:
        for _, loader in self.iter_loaders():
            yield loader

    def iter_loaders(self) -> Iterator[tuple[Hashable, LoaderBase]]:
        """Iterate over all loaders with its image ID."""
        ldr = self._get_loader()
        for key, _, mole in ldr._iter_image_molecules():
            yield key, ldr._make_loader(key, mole)

    def __len__(self) -> int:
        """Number of loaders."""
        ldr = self._get_loader()
        return len(ldr._images) + len(ldr._sources)

    def _get_loader(self) -> BatchLoader:
        loader = self._loader()
        if loader is None:
            raise RuntimeError("Loader is no longer available.")
        return loader
