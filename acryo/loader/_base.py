# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from contextlib import contextmanager
from abc import ABC, abstractmethod, abstractproperty
from typing import (
    Callable,
    TYPE_CHECKING,
    Iterable,
    Iterator,
    NamedTuple,
    Any,
    Sequence,
    SupportsIndex,
    Union,
    overload,
)
from typing_extensions import TypeGuard
import numpy as np
from numpy.typing import NDArray
from dask import array as da
from dask.delayed import delayed
from scipy.spatial.transform import Rotation
import polars as pl

from acryo.alignment import (
    BaseAlignmentModel,
    ZNCCAlignment,
    RotationImplemented,
    AlignmentFactory,
    AlignmentResult,
)
from acryo import _utils
from acryo._types import nm, pixel
from acryo.molecules import Molecules
from acryo.loader import _misc
from acryo.loader._group import LoaderGroup
from acryo.loader._cache import SubtomogramCache
from acryo.pipe._classes import ImageProvider, ImageConverter

if TYPE_CHECKING:
    from typing_extensions import Self
    from dask.delayed import Delayed
    from acryo.classification import PcaClassifier
    from acryo.alignment._base import MaskType

    _ShapeType = Union[pixel, tuple[pixel, ...], None]

TemplateInputType = Union[NDArray[np.float32], ImageProvider]
MaskInputType = Union[NDArray[np.float32], ImageProvider, ImageConverter, None]
AggFunction = Callable[[NDArray[np.float32]], Any]


class Unset:
    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "Unset"


CACHE = SubtomogramCache()


class LoaderBase(ABC):
    """The base class for subtomogram loaders."""

    def __init__(
        self,
        order: int = 3,
        scale: nm = 1.0,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
    ) -> None:
        ndim = 3
        self._order, self._output_shape, self._scale, self._corner_safe = check_input(
            order, output_shape, scale, corner_safe, ndim
        )

    _CACHE = CACHE

    @abstractproperty
    def molecules(self) -> Molecules:
        """All the molecules"""
        raise NotImplementedError

    @property
    def features(self) -> pl.DataFrame:
        """Collect all the features from all the molecules"""
        return self.molecules.features

    @property
    def scale(self) -> nm:
        """Get the physical scale of tomogram."""
        return self._scale

    @property
    def output_shape(self) -> tuple[pixel, pixel, pixel] | Unset:
        """Return the output subtomogram shape."""
        return self._output_shape

    @property
    def order(self) -> int:
        """Return the interpolation order."""
        return self._order

    @property
    def corner_safe(self) -> bool:
        """Whether rotation is corner-safe."""
        return self._corner_safe

    @abstractmethod
    def construct_loading_tasks(
        self, output_shape: _ShapeType = None
    ) -> list[da.Array]:
        ...

    def _get_cached_array(self, shape: tuple[int, int, int] | None) -> da.Array | None:
        return CACHE.get_cache(id(self), shape)

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
        output_shape = self._get_output_shape(output_shape)

        if (cached := self._get_cached_array(output_shape)) is not None:
            return cached

        tasks = self.construct_loading_tasks(output_shape=output_shape)
        out = da.stack(tasks, axis=0)
        return out

    @abstractmethod
    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset | None = None,
        order: int | None = None,
        scale: float | None = None,
        corner_safe: bool | None = None,
    ) -> Self:
        ...

    def copy(self) -> Self:
        """Return a copy of the loader."""
        return self.replace()

    def count(self) -> int:
        """Return the number of subtomograms."""
        return len(self.molecules)

    def iter_mapping_tasks(
        self,
        func: Callable,
        *const_args,
        output_shape: _ShapeType = None,
        var_kwarg: dict[str, Iterable[Any]] | None = None,
        **const_kwargs,
    ) -> Iterator[Delayed]:
        """
        Iterate over delayed mapping tasks using subtomograms.

        Parameters
        ----------
        func : Callable
            Mapping function. The first argument of the function is the subtomogram.
        output_shape : int or tuple of int, optional
            Shape of subtomograms.
        var_kwarg : dict, optional
            Variable keyword arguments. The length of each argument must be the same
            as the number of subtomograms.

        Yields
        ------
        da.delayed.Delayed object
            Delayed tasks that are ready for ``da.compute``.
        """
        output_shape = self._get_output_shape(output_shape)
        dask_array = self.construct_loading_tasks(output_shape=output_shape)
        delayed_f = delayed(func)
        if var_kwarg is None:
            it = (delayed_f(ar, *const_args, **const_kwargs) for ar in dask_array)
        else:
            it = (
                delayed_f(ar, *const_args, **const_kwargs, **kw)
                for ar, kw in zip(dask_array, _misc.dict_iterrows(var_kwarg))
            )
        return it

    def construct_mapping_tasks(
        self,
        func: Callable,
        *const_args,
        output_shape: _ShapeType = None,
        var_kwarg: dict[str, Iterable[Any]] | None = None,
        **const_kwargs,
    ) -> list[Delayed]:
        """
        Construct delayed mapping tasks using subtomograms.

        Parameters
        ----------
        func : Callable
            Mapping function. The first argument of the function is the subtomogram.
        output_shape : int or tuple of int, optional
            Shape of subtomograms.
        var_kwarg : dict, optional
            Variable keyword arguments. The length of each argument must be the same
            as the number of subtomograms.

        Returns
        -------
        list of da.delayed.Delayed object
            List of delayed tasks that are ready for ``da.compute``.
        """
        return list(
            self.iter_mapping_tasks(
                func,
                *const_args,
                output_shape=output_shape,
                var_kwarg=var_kwarg,
                **const_kwargs,
            )
        )

    def _create_cache(self, output_shape: _ShapeType = None) -> da.Array:
        """
        Create cached stack of subtomograms.

        Parameters
        ----------
        path : str, optional
            File path of the temporary file. If not given file will be created by
            ``tempfile.NamedTemporaryFile`` function.

        Returns
        -------
        da.Array
            A lazy-loading array that uses the memory-mapped array.
        """
        output_shape = self._get_output_shape(output_shape)
        if (cached := self._get_cached_array(output_shape)) is not None:
            return cached
        dsk = self.construct_dask(output_shape=output_shape)
        return CACHE.cache_array(dsk, id(self))

    @contextmanager
    def cached(self, output_shape: _ShapeType = None) -> Iterator[da.Array]:
        """
        Context manager for caching subtomograms of give shape.

        Subtomograms are cached in a temporary memory-map file. Within this context
        loading subtomograms of given output shape will be faster.
        """
        with CACHE.temporal():
            yield self._create_cache(output_shape=output_shape)

    def asnumpy(self, output_shape: _ShapeType = None) -> NDArray[np.float32]:
        """Load all the subtomograms as a 4D numpy array."""
        return self.construct_dask(output_shape=output_shape).compute()

    def load(
        self,
        idx: SupportsIndex | slice | Iterable[SupportsIndex],
        output_shape: _ShapeType = None,
    ):
        """
        Load subtomogram(s) of given index.

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
        tasks = self.construct_loading_tasks(output_shape=output_shape)
        if isinstance(idx, SupportsIndex):
            return tasks[idx].compute()
        elif isinstance(idx, slice):
            return da.stack(tasks[idx], axis=0).compute()
        elif hasattr(idx, "__iter__"):
            return da.stack([tasks[i] for i in idx], axis=0).compute()
        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def load_iter(
        self, output_shape: _ShapeType = None
    ) -> Iterator[NDArray[np.float32]]:
        """
        Iterate over subtomograms.

        Parameters
        ----------
        output_shape : int or tuple of int, optional
            Shape of the output subtomograms.

        Yields
        ------
        3D array
            Subtomogram
        """
        tasks = self.construct_loading_tasks(output_shape=output_shape)
        for task in tasks:
            yield task.compute()

    def average(self, output_shape: _ShapeType = None) -> NDArray[np.float32]:
        """
        Calculate the average of subtomograms.

        This function execute so-called "subtomogram averaging". The size of
        subtomograms is determined by the ``self.output_shape`` attribute.

        Returns
        -------
        np.ndarray
            Averaged image
        """
        output_shape = self._get_output_shape(output_shape)
        dsk = self.construct_dask(output_shape=output_shape)
        dsk = dsk.rechunk(("auto",) + output_shape)  # type: ignore
        return da.compute(da.mean(dsk, axis=0))[0]

    def average_split(
        self,
        n_set: int = 1,
        seed: int | None = 0,
        squeeze: bool = True,
        output_shape: _ShapeType = None,
    ) -> NDArray[np.float32]:
        """
        Split subtomograms into two set and average separately.

        This method executes pairwise subtomogram averaging using randomly
        selected molecules, which is useful for calculation of such as Fourier
        shell correlation.

        Parameters
        ----------
        n_set : int, default is 1
            Number of split set of averaged image.
        seed : random seed, default is 0
            Random seed to determine how subtomograms will be split.
        squeeze : bool, default is True
            If true and n_set is 1, return a 4D array.
        output_shape : tuple of int, optional
            Output shape of the averaged image. If not given, the default output
            shape of the loader object will be used.

        Returns
        -------
        np.ndarray
            Averaged images. The shape of the array is (n_set, 2, *output_shape).
        """
        output_shape = self._get_output_shape(output_shape)
        rng = np.random.default_rng(seed=seed)

        tasks: list[da.Array] = []
        dsk = self.construct_dask(output_shape=output_shape)
        nmole = dsk.shape[0]
        for _ in range(n_set):
            ind0, ind1 = _misc.random_splitter(rng, nmole)
            _stack = da.stack(
                [
                    da.mean(dsk[ind0].rechunk(("auto",) + output_shape), axis=0),  # type: ignore
                    da.mean(dsk[ind1].rechunk(("auto",) + output_shape), axis=0),  # type: ignore
                ],
                axis=0,
            )
            tasks.append(_stack)

        out = da.compute(tasks)[0]
        stack = np.stack(out, axis=0)
        if squeeze and n_set == 1:
            stack = stack[0]
        return stack

    def align(
        self,
        template: TemplateInputType,
        *,
        mask: MaskInputType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] | AlignmentFactory = ZNCCAlignment,
        **align_kwargs,
    ) -> Self:
        """
        Align subtomograms to the template image.

        This method conduct so called "subtomogram alignment". Only shifts and rotations
        are calculated in this method. To get averaged image, you'll have to run "average"
        method using the resulting SubtomogramLoader instance.

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
        if np.isscalar(max_shifts):
            max_shifts = (max_shifts,) * 3
        _max_shifts_px = np.asarray(max_shifts) / self.scale

        model = alignment_model(
            template=self.normalize_template(template),
            mask=self.normalize_mask(mask),
            **align_kwargs,
        )
        tasks = self.construct_mapping_tasks(
            model.align,
            max_shifts=_max_shifts_px,
            output_shape=model.input_shape,
            var_kwarg=dict(
                quaternion=self.molecules.quaternion(),
                pos=self.molecules.pos / self.scale,
            ),
        )
        all_results = da.compute(tasks)[0]
        return self._post_align(all_results, model.input_shape)

    def _post_align(
        self,
        results: list[AlignmentResult],
        shape: tuple[int, int, int],
    ) -> Self:
        local_shifts, local_rot, scores = _misc.allocate(len(results))
        for i, result in enumerate(results):
            _, local_shifts[i], local_rot[i], scores[i] = result

        rotator = Rotation.from_quat(local_rot)
        mole_aligned = self.molecules.linear_transform(
            local_shifts * self.scale,
            rotator,
        )

        mole_aligned.features = self.molecules.features.with_columns(
            _misc.get_feature_list(scores, local_shifts, rotator.as_rotvec()),
        )

        return self.replace(molecules=mole_aligned, output_shape=shape)

    def align_no_template(
        self,
        *,
        mask: MaskInputType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        output_shape: _ShapeType = None,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Self:
        """
        Align subtomograms without template image.

        A template-free version of :func:`align`. This method first
        calculates averaged image and uses it for the alignment template. To
        avoid loading same subtomograms twice, a memory-mapped array is created
        internally (so the second subtomogram loading is faster).

        Parameters
        ----------
        mask : np.ndarray or callable, optional
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
        if output_shape is None and isinstance(mask, np.ndarray):
            output_shape = mask.shape

        with self.cached(output_shape=output_shape):
            template = self.average(output_shape=output_shape)
            out = self.align(
                template,
                mask=self.normalize_mask(mask),
                max_shifts=max_shifts,
                alignment_model=alignment_model,
                **align_kwargs,
            )
        return out

    def align_multi_templates(
        self,
        templates: list[TemplateInputType],
        *,
        mask: MaskInputType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        label_name: str = "labels",
        **align_kwargs,
    ) -> Self:
        """
        Align subtomograms with multiple template images.

        A multi-template version of :func:`align`. This method calculate cross
        correlation for every template and uses the best local shift, rotation and
        template.

        Parameters
        ----------
        templates: list of 3D arrays or ImageProvider
            Template images. If ImageProvider is given, the image will be provided
            accordingly using the scale of the loader object.
        mask : np.ndarray, optional
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

        _max_shifts_px = np.asarray(max_shifts) / self.scale

        model = alignment_model(
            template=[self.normalize_template(t) for t in templates],
            mask=self.normalize_mask(mask),
            **align_kwargs,
        )
        tasks = self.construct_mapping_tasks(
            model.align,
            max_shifts=_max_shifts_px,
            output_shape=model.input_shape,
            var_kwarg=dict(
                quaternion=self.molecules.quaternion(),
                pos=self.molecules.pos / self.scale,
            ),
        )
        all_results = da.compute(tasks)[0]
        if isinstance(model, RotationImplemented) and model._n_rotations > 1:
            remainder = len(templates)
        else:
            remainder = -1
        return self._post_align_multi_templates(
            all_results, model.input_shape, remainder, label_name
        )

    def _post_align_multi_templates(
        self,
        results: list[AlignmentResult],
        shape: tuple[int, int, int],
        remainder: int = -1,
        label_name: str = "labels",
    ):

        local_shifts, local_rot, scores = _misc.allocate(len(results))
        labels = np.zeros(len(results), dtype=np.uint32)
        for i, result in enumerate(results):
            labels[i], local_shifts[i], local_rot[i], scores[i] = result

        rotator = Rotation.from_quat(local_rot)
        mole_aligned = self.molecules.linear_transform(
            local_shifts * self.scale,
            rotator,
        )

        if remainder > 1:
            labels %= remainder
        labels = labels.astype(np.uint8)

        feature_list = _misc.get_feature_list(scores, local_shifts, rotator.as_rotvec())
        mole_aligned.features = self.molecules.features.with_columns(
            feature_list + [pl.Series(label_name, labels)]
        )

        return self.replace(molecules=mole_aligned, output_shape=shape)

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
        """
        Construct a dask array of correlation landscape.

        This method internally calls the ``landscape`` method of the input alignment
        model.
        """
        if np.isscalar(max_shifts):
            max_shifts = (max_shifts,) * 3
        _max_shifts_px = tuple(np.asarray(max_shifts) / self.scale)

        model = alignment_model(
            template=self.normalize_template(template),
            mask=self.normalize_mask(mask),
            **align_kwargs,
        )

        task_shape = 2 * np.ceil(_max_shifts_px) + 1
        task_arrays: list[da.Array] = [
            da.from_delayed(task, shape=tuple(task_shape), dtype=np.float32)
            for task in self.iter_mapping_tasks(
                model.landscape,
                max_shifts=_max_shifts_px,
                upsample=upsample,
                var_kwarg=dict(
                    quaternion=self.molecules.quaternion(),
                    pos=self.molecules.pos / self.scale,
                )
            )
        ]
        return da.stack(task_arrays, axis=0)

    def apply(
        self,
        func: AggFunction | Iterable[AggFunction],
        schema: list[str] | None = None,
    ) -> pl.DataFrame:
        """
        Apply functions to subtomograms.

        Parameters
        ----------
        funcs : callable or list of callable
            Functions that take a subtomogram as input and return a scalar.
        schema : list[str], optional
            DataFrame schema.

        Returns
        -------
        pl.DataFrame
            Result table.
        """
        all_tasks: list[list[Delayed]] = []
        if _is_iterable_of_funcs(func):
            funcs = func
        else:
            funcs: Iterable[AggFunction] = [func]  # type: ignore
        if schema is None:
            schema = [fn.__name__ for fn in funcs]
        if len(set(schema)) != len(schema):
            raise ValueError("Schema names must be unique.")
        if isinstance(self.output_shape, Unset):
            raise ValueError("Output shape is unknown.")
        for fn in funcs:
            tasks = self.construct_mapping_tasks(fn, output_shape=self.output_shape)
            all_tasks.append(tasks)
        all_results = np.array(da.compute(all_tasks)[0])
        return pl.DataFrame(all_results, schema=schema)

    def fsc(
        self,
        mask: TemplateInputType | None = None,
        seed: int | None = 0,
        n_set: int = 1,
        dfreq: float = 0.05,
    ) -> pl.DataFrame:
        """
        Calculate Fourier shell correlation.

        Parameters
        ----------
        mask : np.ndarray or ImageProvider, optional
            Mask image
        seed : random seed, default is 0
            Random seed used to split subtomograms.
        n_set : int, default is 1
            Number of split set of averaged images.
        dfreq : float, optional
            Frequency sampling width. Automatically determined if not provided.

        Returns
        -------
        pl.DataFrame
            A data frame with FSC results.

        See Also
        --------
        fsc_with_average
        """
        return self.fsc_with_average(mask, seed, n_set, dfreq)[0]

    def fsc_with_average(
        self,
        mask: TemplateInputType | None = None,
        seed: int | None = 0,
        n_set: int = 1,
        dfreq: float | None = None,
    ) -> tuple[pl.DataFrame, NDArray[np.float32]]:
        """
        Calculate Fourier shell correlation and the resulting averaged image.

        Parameters
        ----------
        mask : np.ndarray or ImageProvider, optional
            Mask image
        seed : random seed, default is 0
            Random seed used to split subtomograms.
        n_set : int, default is 1
            Number of split set of averaged images.
        dfreq : float, optional
            Frequency sampling width. Automatically determined if not provided.

        Returns
        -------
        pl.DataFrame and 3D array
            A data frame with FSC results and the average image. Data frame has
            columns "freq", "FSC-0", ..., "FSC-{n-1}" where n is the number of
            split.

        See Also
        --------
        fsc
        """
        if mask is None:
            _mask = 1.0
            output_shape = self.output_shape
            if isinstance(output_shape, Unset):
                raise TypeError("Output shape is unknown.")
        elif isinstance(mask, ImageProvider):
            _mask = mask(self.scale)
            output_shape = _mask.shape
        else:
            _mask = mask
            output_shape = mask.shape

        if n_set <= 0:
            raise ValueError("'n_set' must be positive.")
        if dfreq is None:
            dfreq = 1.5 / min(output_shape)
        img = self.average_split(
            n_set=n_set,
            seed=seed,
            squeeze=False,
            output_shape=output_shape,
        )
        img[:] -= img.mean()  # normalize to minimize the sharp mask edge effect
        fsc_all: dict[str, np.ndarray] = {}
        freq = np.zeros(0, dtype=np.float32)
        for i in range(n_set):
            img0, img1 = img[i]
            freq, fsc = _utils.fourier_shell_correlation(
                img0 * _mask, img1 * _mask, dfreq=dfreq
            )
            fsc_all[f"FSC-{i}"] = fsc

        out: dict[str, NDArray[np.float32]] = {"freq": freq}
        out.update(fsc_all)
        return pl.DataFrame(out), np.mean(img[0], axis=0)

    def classify(
        self,
        template: TemplateInputType | None = None,
        mask: MaskInputType = None,
        *,
        cutoff: float = 0.5,
        n_components: int = 2,
        n_clusters: int = 2,
        tilt_range: tuple[float, float] | None = None,
        seed: int = 0,
        label_name: str = "cluster",
    ) -> ClassificationResult:
        """
        Classify 3D densities by PCA of wedge-masked differences.

        Parameters
        ----------
        template : 3D array, optional
            Template image to calculate the difference. If not given, average image will
            be used.
        mask : 3D array, optional
            Soft mask of the same shape as the template or the output_shape parameter.
        n_components : int, default is 2
            Number of PCA components.
        n_clusters : int, default is 2
            Number of classes.
        tilt_range : (float, float), optional
            Tilt range in degree.
        seed : int, default is 0
            Random seed used for K-means clustering.
        label_name : str, default is "cluster"
            Column name used for the output classes.

        Returns
        -------
        ClassificationResult
            Tuple of SubtomogramLoader and PCA classifier object.

        References
        ----------
        - Heumann, J. M., Hoenger, A., & Mastronarde, D. N. (2011). Clustering and variance
          maps for cryo-electron tomography using wedge-masked differences. Journal of
          structural biology, 175(3), 288-299.
        """
        from acryo.classification import PcaClassifier

        if template is not None:
            template = self.normalize_template(template)
        _mask = self.normalize_mask(mask)
        if isinstance(_mask, np.ndarray):
            shape = _mask.shape
        elif isinstance(template, np.ndarray):
            shape = template.shape
        elif isinstance(self.output_shape, Unset):
            raise ValueError("Cannot determine output shape.")
        else:
            shape = self.output_shape

        with self.cached(shape):
            if template is None:
                template = self.average(shape)
            model = ZNCCAlignment(template, _mask, cutoff=cutoff, tilt_range=tilt_range)
            tasks: list[da.Array] = []
            for task in self.iter_mapping_tasks(
                model.masked_difference,
                output_shape=shape,
                var_kwarg=dict(quaternion=self.molecules.quaternion()),
            ):
                tasks.append(da.from_delayed(task, shape=shape, dtype=np.float32))

            # PCA requires aggregation along the first axis.
            # Rechunk to improve performance.
            stack = da.stack(tasks, axis=0).rechunk(("auto",) + shape)  # type: ignore

            clf = PcaClassifier(
                stack,
                model.mask,
                n_components=n_components,
                n_clusters=n_clusters,
                seed=seed,
            )
            clf.run()

        mole = self.molecules.copy()
        mole.features = mole.features.with_columns(pl.Series(label_name, clf._labels))
        new = self.replace(molecules=mole)
        return ClassificationResult(new, clf)

    def reshape(
        self,
        template: TemplateInputType | None = None,
        mask: MaskInputType = None,
        shape: tuple[int, int, int] | None = None,
    ) -> Self:
        """Return a new loader with appropriate output shape."""
        shapes: set[tuple[int, int, int]] = set()
        if template is not None:
            shapes.add(self.normalize_template(template).shape)
        if mask is not None:
            if isinstance(mask, np.ndarray):
                shapes.add(mask.shape)
            elif isinstance(mask, ImageProvider):
                shapes.add(mask(self.scale).shape)
        if shape is not None:
            shapes.add(shape)
        if len(shapes) == 0:
            raise ValueError("Cannot infer shape from the input.")
        elif len(shapes) > 1:
            raise ValueError("Inconsistent shapes.")
        return self.replace(output_shape=shapes.pop())

    def head(self, n: int = 10) -> Self:
        """Return a new loader with the first n molecules."""
        return self.replace(molecules=self.molecules.head(n))

    def tail(self, n: int = 10) -> Self:
        """Return a new loader with the last n molecules."""
        return self.replace(molecules=self.molecules.tail(n))

    def sample(self, n: int = 10, seed: int | None = None) -> Self:
        """Return a new loader with randomly sampled molecules."""
        return self.replace(molecules=self.molecules.sample(n, seed))

    def filter(
        self,
        predicate: pl.Expr | str | pl.Series | list[bool] | np.ndarray,
    ) -> Self:
        """Return a new loader with filtered molecules."""
        out = self.replace(molecules=self.molecules.filter(predicate))
        if (cached := self._get_cached_array(shape=None)) is not None:
            if isinstance(predicate, (str, pl.Expr)):
                sl = self.molecules.features.select(predicate).to_numpy().ravel()
            else:
                sl = np.asarray(predicate)
            CACHE.cache_array(cached[sl], id(out))
        return out

    @overload
    def groupby(self, by: str | pl.Expr) -> LoaderGroup[str, Self]:
        ...

    @overload
    def groupby(
        self, by: Sequence[str | pl.Expr]
    ) -> LoaderGroup[tuple[str, ...], Self]:
        ...

    def groupby(self, by):
        """
        Group loader by given feature column(s).

        >>> for key, loader in loader.groupby("score"):
        ...     print(key, loader)
        """
        if not isinstance(by, (str, pl.Expr)):
            by = tuple(by)
        return LoaderGroup._from_loader(self, by)

    def _get_output_shape(self, output_shape: _ShapeType) -> tuple[pixel, ...]:
        if output_shape is None:
            if isinstance(self.output_shape, Unset):
                raise ValueError("Output shape is unknown.")
            _output_shape = self.output_shape
        else:
            _output_shape = _misc.normalize_shape(output_shape, ndim=3)
        return _output_shape

    def normalize_template(self, template: TemplateInputType) -> NDArray[np.float32]:
        """Resolve any template input type to a 3D array."""
        if isinstance(template, np.ndarray):
            return template
        elif isinstance(template, ImageProvider):
            out = template(self.scale)
            return np.asarray(out, dtype=np.float32)
        raise TypeError(f"Invalid template type: {type(template)}")

    def normalize_mask(self, mask: MaskInputType) -> MaskType:
        """Resolve any mask input type to a 3D array."""
        if isinstance(mask, (np.ndarray, type(None))):
            return mask
        elif isinstance(mask, ImageProvider):
            out = mask(self.scale)
            return np.asarray(out, dtype=np.float32)
        elif isinstance(mask, ImageConverter):
            return mask.with_scale(self.scale)
        raise TypeError(f"Invalid mask type: {type(mask)}")

    @overload
    def normalize_input(
        self,
        template: None = None,
        mask: MaskInputType = None,
    ) -> tuple[None, NDArray[np.float32] | None]:
        ...

    @overload
    def normalize_input(
        self,
        template: TemplateInputType,
        mask: MaskInputType = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]:
        ...

    def normalize_input(self, template=None, mask=None):
        """Resolve any template and mask input types to 3D arrays."""
        if template is not None:
            _template = self.normalize_template(template)
            _mask = self.normalize_mask(mask)
            if callable(_mask):
                _mask = _mask(_template)
            return _template, _mask
        else:
            _mask = self.normalize_mask(mask)
            if callable(mask):
                raise TypeError("Cannot determine mask array without template.")
            return None, _mask


class ClassificationResult(NamedTuple):
    """Tuple of classification results."""

    loader: LoaderBase
    classifier: PcaClassifier


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
    elif output_shape is None:
        _output_shape = Unset()
    else:
        _output_shape = _misc.normalize_shape(output_shape, ndim=ndim)

    # check scale
    _scale = float(scale)
    if _scale <= 0:
        raise ValueError("Negative scale is not allowed.")

    _corner_safe = bool(corner_safe)
    return order, _output_shape, _scale, _corner_safe


def _is_iterable_of_funcs(x: Any) -> TypeGuard[Iterable[AggFunction]]:
    if not hasattr(x, "__iter__"):
        return False
    return all(callable(f) for f in x)
