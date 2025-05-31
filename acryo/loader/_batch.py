# pyright: reportPrivateImportUsage=false

from __future__ import annotations
from types import MappingProxyType

import weakref
from typing import Hashable, Iterable, Iterator, TYPE_CHECKING
import numpy as np
import polars as pl

from dask import array as da

from acryo import _utils
from acryo.alignment._base import TomographyInput
from acryo.molecules import Molecules
from acryo.backend import Backend
from acryo._types import nm, pixel
from acryo._dask import DaskArrayList, DaskTaskList
from acryo.loader import _misc
from acryo.loader._base import LoaderBase, MaskInputType, TemplateInputType
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
    """

    def __init__(
        self,
        order: int = 3,
        scale: nm = 1.0,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
    ) -> None:
        super().__init__(order, scale, output_shape, corner_safe)
        self._images: dict[Hashable, NDArray[np.float32] | da.Array] = {}
        self._molecules: Molecules = Molecules.empty([IMAGE_ID_LABEL])
        self._tilt_models: dict[Hashable, TomographyInput] = {}

    def __repr__(self) -> str:
        loaders_repr = []
        for loader in self.loaders:
            shape = loader.image.shape
            mole_repr = repr(loader.molecules)
            loaders_repr.append(
                f"{loader.__class__.__name__}(tomogram={shape}, "
                f"molecules={mole_repr})"
            )
        loaders_repr = ", ".join(loaders_repr)
        return (
            f"{self.__class__.__name__}(loaders=[{loaders_repr}], "
            f"output_shape={self.output_shape}, order={self.order}, "
            f"scale={self.scale:.4f})"
        )

    @property
    def loaders(self) -> LoaderAccessor:
        """Interface to access the subtomogram loaders."""
        return LoaderAccessor(self)

    @property
    def images(self) -> MappingProxyType:
        """All the images in the collection."""
        return MappingProxyType(self._images)

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
        if image_id is None:
            image_id = len(self._images)
            while image_id in self._images:
                image_id += 1
        molecules = molecules.copy()
        molecules.features = molecules.features.with_columns(
            pl.Series(IMAGE_ID_LABEL, np.full((len(molecules)), image_id))
        )
        _molecules_new = self._molecules.concat_with(molecules)

        self._images[image_id] = image
        self._tilt_models[image_id] = tilt_model
        self._molecules = _molecules_new
        return self

    def add_loader(self, loader: SubtomogramLoader | BatchLoader) -> Self:
        """Add a subtomogram loader or a batch loader to the collection."""
        if isinstance(loader, SubtomogramLoader):
            image = loader.image
            molecules = loader.molecules
            self.add_tomogram(image, molecules, tilt_model=loader.tilt_model)
        elif isinstance(loader, BatchLoader):
            for sub in loader.loaders:
                self.add_loader(sub)
        else:
            raise TypeError(f"Cannot add {type(loader)}.")
        return self

    @classmethod
    def from_loaders(
        cls,
        loaders: Iterable[SubtomogramLoader],
        order: int = 3,
        scale: nm = 1.0,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
    ) -> Self:
        """Construct a loader from a list of loaders."""
        self = cls(
            order=order,
            scale=scale,
            output_shape=output_shape,
            corner_safe=corner_safe,
        )
        for loader in loaders:
            self.add_loader(loader)
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
        corner_safe: bool | None = None,
    ) -> Self:
        """Return a new instance with different parameter(s)."""
        if output_shape is None:
            output_shape = self.output_shape
        if order is None:
            order = self.order
        if scale is None:
            scale = self.scale
        if corner_safe is None:
            corner_safe = self.corner_safe
        out = type(self)(
            order=order,
            scale=scale,
            output_shape=output_shape,
        )
        out._images = self._images.copy()
        out._tilt_models = self._tilt_models.copy()
        if molecules is None:
            out._molecules = self._molecules.copy()
        else:
            out._molecules = molecules
            _id_exists = set(molecules.features[IMAGE_ID_LABEL].unique())
            for k in list(out._images.keys()):
                if k not in _id_exists:
                    out._images.pop(k)
        return out

    def binning(self, binsize: int, *, compute: bool = False) -> Self:
        """Return a new instance with binned images."""
        if binsize == 1:
            return self.copy()
        tr = -(binsize - 1) / 2 * self.scale
        molecules = self.molecules.translate([tr, tr, tr])
        _images: dict[Hashable, NDArray[np.float32] | da.Array] = {}
        _compute_dict: dict[Hashable, da.Array] = {}
        for _id, image in self._images.items():
            binned_image = _utils.bin_image(image, binsize=binsize)
            if isinstance(binned_image, da.Array) and compute:
                _compute_dict[_id] = binned_image
            _images[_id] = binned_image

        if _compute_dict:
            _computed = da.compute(_compute_dict)
            _images.update(_computed)

        out = self.replace(
            molecules=molecules,
            scale=self.scale * binsize,
        )
        out._images = _images
        return out

    def construct_loading_tasks(
        self,
        output_shape: _ShapeType = None,
        backend: Backend | None = None,
    ) -> DaskArrayList:
        """Construct batch loading tasks."""
        _backend = backend or Backend()
        return DaskArrayList.concat(
            loader.construct_loading_tasks(output_shape=output_shape, backend=_backend)
            for loader in self.loaders
        )

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
                remainder = model.remainder()
                tasks.extend(
                    each._prep_align_multi_templates(model, max_shifts, _backend)
                )
            else:
                tasks.extend(each._prep_align_tasks(model, max_shifts, _backend))

        all_results = tasks.compute()

        if is_multi:
            return self._post_align_multi_templates(
                all_results, model.input_shape, remainder
            )
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
        return da.concatenate(arrays, axis=0)

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
        return np.concatenate(scores, axis=1)

    def _prep_classify_stack(
        self,
        template: TemplateInputType,
        mask: MaskInputType,
        cutoff: float = 1.0,
        shape: tuple[int, int, int] | None = None,
    ):
        stacks = []
        for each in self.loaders:
            stacks.append(each._prep_classify_stack(template, mask, cutoff, shape))
        return da.concatenate(stacks, axis=0)


class LoaderAccessor:
    """The interface to access subtomogram loaders in a BatchLoader."""

    def __init__(self, collection: BatchLoader):
        self._loader = weakref.ref(collection)

    def __getitem__(self, idx: int) -> SubtomogramLoader:
        ldr = self._get_loader()
        mole = ldr.filter(pl.col(IMAGE_ID_LABEL) == idx).molecules
        if mole.features.shape[0] == 0:
            raise KeyError(idx)
        image = ldr._images[idx]
        loader = SubtomogramLoader(
            image,
            mole,
            order=ldr.order,
            scale=ldr.scale,
            output_shape=ldr.output_shape,
            tilt_model=ldr._tilt_models.get(idx, None),
        )
        return loader

    def __iter__(self) -> Iterator[SubtomogramLoader]:
        ldr = self._get_loader()
        for key, group in ldr.molecules.groupby(IMAGE_ID_LABEL):
            image = ldr._images[key]
            tilt_model = ldr._tilt_models.get(key, None)
            loader = SubtomogramLoader(
                image,
                group,
                ldr.order,
                ldr.scale,
                ldr.output_shape,
                tilt_model=tilt_model,
                corner_safe=ldr.corner_safe,
            )
            yield loader

    def __len__(self) -> int:
        """Number of loaders."""
        ldr = self._get_loader()
        return len(ldr._images)

    def _get_loader(self) -> BatchLoader:
        loader = self._loader()
        if loader is None:
            raise RuntimeError("Loader is no longer available.")
        return loader
