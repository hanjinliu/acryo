# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Callable,
    TYPE_CHECKING,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Any,
    Sequence,
    SupportsIndex,
    TypeVar,
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
    AlignmentFactory,
    AlignmentResult,
)
from acryo import _utils
from acryo._types import nm, pixel
from acryo.backend import Backend
from acryo._dask import DaskArrayList, DaskTaskList, DaskTaskIterator, compute
from acryo.molecules import Molecules
from acryo.loader import _misc
from acryo.loader._group import LoaderGroup
from acryo.pipe._classes import ImageProvider, ImageConverter

if TYPE_CHECKING:
    from typing_extensions import Self
    from acryo.classification import PcaClassifier
    from acryo.alignment._base import MaskType

_R = TypeVar("_R")

_ShapeType = Union[pixel, tuple[pixel, ...], None]
TemplateInputType = Union[NDArray[np.float32], ImageProvider[np.ndarray]]
MaskInputType = Union[
    NDArray[np.float32], ImageProvider[np.ndarray], ImageConverter, None
]
AggFunction = Callable[[NDArray[np.float32]], _R]
IntoExpr = Union[str, pl.Expr]


class Unset:
    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "Unset"


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
        self._order, self._output_shape, self._scale = check_input(
            order, output_shape, scale, ndim
        )
        self._corner_safe = corner_safe

    @property
    @abstractmethod
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
        return self._output_shape  # type: ignore

    @property
    def order(self) -> int:
        """Return the interpolation order."""
        return self._order

    @property
    def corner_safe(self) -> bool:
        """Return whether the loader is corner-safe."""
        return self._corner_safe

    @abstractmethod
    def construct_loading_tasks(
        self,
        output_shape: _ShapeType = None,
        backend: Backend | None = None,
    ) -> DaskArrayList[np.number]: ...

    def construct_dask(
        self,
        output_shape: pixel | tuple[pixel, ...] | None = None,
        backend: Backend | None = None,
    ) -> da.Array:
        """Construct a dask array of subtomograms.

        This function is always needed before parallel processing.

        Returns
        -------
        da.Array
            An 4-D array which ``arr[i]`` corresponds to the ``i``-th subtomogram.
        """
        output_shape = self._get_output_shape(output_shape)
        xp = backend or Backend()
        tasks = self.construct_loading_tasks(output_shape, xp)
        out = da.stack(tasks, axis=0)
        return out

    @abstractmethod
    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset | None = None,
        order: int | None = None,
        scale: float | None = None,
    ) -> Self:
        """Return a new loader with updated attributes."""

    def copy(self) -> Self:
        """Return a copy of the loader."""
        return self.replace()

    def count(self) -> int:
        """Return the number of subtomograms."""
        return len(self.molecules)

    def iter_mapping_tasks(
        self,
        func: Callable[..., _R],
        *const_args,
        output_shape: _ShapeType = None,
        var_kwarg: dict[str, Iterable[Any]] | None = None,
        **const_kwargs,
    ) -> DaskTaskIterator[_R]:
        """Iterate over delayed mapping tasks using subtomograms.

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
        return DaskTaskIterator(it)

    def construct_mapping_tasks(
        self,
        func: Callable[..., _R],
        *const_args,
        output_shape: _ShapeType = None,
        var_kwarg: dict[str, Iterable[Any]] | None = None,
        **const_kwargs,
    ) -> DaskTaskList[_R]:
        """Construct delayed mapping tasks using subtomograms.

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
        DaskTaskList
            List of delayed tasks that are ready for computation.
        """
        return self.iter_mapping_tasks(
            func,
            *const_args,
            output_shape=output_shape,
            var_kwarg=var_kwarg,
            **const_kwargs,
        ).tolist()

    def asnumpy(self, output_shape: _ShapeType = None) -> NDArray[np.float32]:
        """Load all the subtomograms as a 4D numpy array."""
        return self.construct_dask(output_shape=output_shape).compute()

    def load(
        self,
        idx: SupportsIndex | slice | Iterable[SupportsIndex],
        output_shape: _ShapeType = None,
    ) -> NDArray[np.float32]:
        """Load subtomogram(s) of given index.

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
        """Iterate over subtomograms.

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

    def average(
        self, output_shape: _ShapeType = None, *, backend: Backend | None = None
    ) -> NDArray[np.float32]:
        """Calculate the average of subtomograms.

        This function execute so-called "subtomogram averaging". The size of
        subtomograms is determined by the ``self.output_shape`` attribute.

        Returns
        -------
        np.ndarray
            Averaged image
        """
        xp = backend or Backend()
        output_shape = self._get_output_shape(output_shape)
        dsk = self.construct_dask(output_shape=output_shape, backend=xp)
        dsk = dsk.rechunk(("auto",) + output_shape)  # type: ignore
        return xp.asnumpy(dsk.mean(axis=0).compute())

    def average_split(
        self,
        n_set: int = 1,
        seed: int | None = 0,
        squeeze: bool = True,
        output_shape: _ShapeType = None,
    ) -> NDArray[np.float32]:
        """Split subtomograms into two set and average separately.

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
        backend = Backend()
        output_shape = self._get_output_shape(output_shape)
        rng = np.random.default_rng(seed=seed)

        tasks: list[da.Array] = []
        dsk = self.construct_dask(output_shape=output_shape)
        nmole = dsk.shape[0]
        chunksize = ("auto",) + output_shape
        for _ in range(n_set):
            ind0, ind1 = _misc.random_splitter(rng, nmole)
            _stack = da.stack(
                [
                    dsk[ind0].rechunk(chunksize).mean(axis=0),  # type: ignore
                    dsk[ind1].rechunk(chunksize).mean(axis=0),  # type: ignore
                ],
                axis=0,
            )
            tasks.append(_stack)

        out = da.compute(tasks)[0]
        stack = np.stack([backend.asnumpy(a) for a in out], axis=0)
        if squeeze and n_set == 1:
            stack: NDArray[np.float32] = stack[0]
        return stack

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
        model = alignment_model(
            self.normalize_template(template, allow_multiple=True),
            self.normalize_mask(mask),
            **self._update_align_kwargs(align_kwargs),
        )
        if model.has_hetero_templates:
            return self.align_multi_templates(
                list(model.template),
                mask=mask,
                max_shifts=max_shifts,
                alignment_model=alignment_model,
                backend=backend,
                **align_kwargs,
            )

        _backend = backend or Backend()
        max_shifts = _misc.normalize_max_shifts(max_shifts)
        tasks = self._prep_align_tasks(model, max_shifts, _backend)
        all_results = tasks.compute()
        return self._post_align(all_results, model.input_shape)

    def align_no_template(
        self,
        *,
        mask: MaskInputType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        output_shape: _ShapeType = None,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Self:
        """Align subtomograms without template image.

        A template-free version of :func:`align`. This method first
        calculates averaged image and uses it for the alignment template. To
        avoid loading same subtomograms twice, a memory-mapped array is created
        internally (so the second subtomogram loading is faster).

        Parameters
        ----------
        mask : 3D array, ImageProvider or ImageConverter, optional
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
        backend = Backend()
        template = self.average(output_shape=output_shape, backend=backend)
        out = self.align(
            template,
            mask=mask,
            max_shifts=max_shifts,
            alignment_model=alignment_model,
            backend=backend,
            **align_kwargs,
        )
        return out

    def align_multi_templates(
        self,
        templates: list[TemplateInputType] | ImageProvider[list[NDArray[np.float32]]],
        *,
        mask: MaskInputType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] | AlignmentFactory = ZNCCAlignment,
        backend: Backend | None = None,
        label_name: str = "labels",
        **align_kwargs,
    ) -> Self:
        """Align subtomograms with multiple template images.

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

        if isinstance(templates, ImageProvider):
            _templates = templates(self.scale)
        else:
            _templates = [self.normalize_template(t) for t in templates]
        model = alignment_model(
            template=_templates,
            mask=self.normalize_mask(mask),
            **self._update_align_kwargs(align_kwargs),
        )
        tasks = self._prep_align_multi_templates(model, max_shifts, backend)
        all_results = tasks.compute()
        remainder = model.remainder()
        return self._post_align_multi_templates(
            all_results, model.input_shape, remainder, label_name
        )

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
        """Construct a dask array of correlation landscape.

        This method internally calls the ``landscape`` method of the input alignment
        model.

        Returns
        -------
        dask array
            If input alignment model has rotation or multiple templates, the output
            shape will be (P, M, Nz, Ny, Nx), otherwise (P, Nz, Ny, Nx), where P is for
            each molecules, M is the number of created template images, and Nz, Ny and
            Nx is the up-sampled shape of search range.
        """
        max_shifts = _misc.normalize_max_shifts(max_shifts)
        _max_shifts_px = tuple(np.asarray(max_shifts) / self.scale)

        model = alignment_model(
            self.normalize_template(template, allow_multiple=True),
            self.normalize_mask(mask),
            **self._update_align_kwargs(align_kwargs),
        )

        if model.is_multi_templates:
            task_shape = (model.niter,) + tuple(
                2 * np.ceil(_max_shifts_px).astype(np.int32) + 1
            )
        else:
            task_shape = tuple(2 * np.ceil(_max_shifts_px).astype(np.int32) + 1)
        task_arrays = (
            self.replace(output_shape=model.input_shape)
            .iter_mapping_tasks(
                model.landscape,
                max_shifts=_max_shifts_px,
                upsample=upsample,
                var_kwarg=dict(
                    quaternion=self.molecules.quaternion(),
                    pos=self.molecules.pos / self.scale,
                ),
            )
            .tolist()
            .asarrays(shape=task_shape, dtype=np.float32)
        )
        return da.stack(task_arrays, axis=0)

    def score(
        self,
        templates: list[TemplateInputType],
        *,
        mask: MaskInputType = None,
        alignment_model: type[BaseAlignmentModel] | AlignmentFactory = ZNCCAlignment,
        **align_kwargs,
    ) -> NDArray[np.float32]:
        """Calculate the score of subtomograms against the template image.

        This method internally calls the ``score`` method of the input alignment model.

        Parameters
        ----------
        templates : list of 3D arrays or ImageProvider
            Template images. If ImageProvider is given, the image will be provided
            accordingly using the scale of the loader object.
        mask : 3D array, ImageProvider or ImageConverter, optional
            Mask image. Must in the same shape as the template.
        alignment_model : subclass of BaseAlignmentModel, optional
            Alignment model class used for subtomogram alignment. By default,
            ``ZNCCAlignment`` will be used.

        Returns
        -------
        np.ndarray
            2D arrays of scores, where score[i, j] is the score of the i-th template
            for the j-th subtomogram.
        """
        tasks: list[DaskTaskList[float]] = []
        for template in templates:
            model = alignment_model(
                self.normalize_template(template),
                self.normalize_mask(mask),
                **self._update_align_kwargs(align_kwargs),
            )
            task = self.construct_mapping_tasks(
                model.score,
                output_shape=model.input_shape,
                var_kwarg=dict(
                    quaternion=self.molecules.quaternion(),
                    pos=self.molecules.pos / self.scale,
                ),
            )
            tasks.append(task)
        out = compute(tasks)
        return np.array(out, dtype=np.float32)

    def apply(
        self,
        func: AggFunction[_R] | Iterable[AggFunction[_R]],
        *more_funcs: AggFunction[_R],
        schema: list[str] | dict[str, type[pl.DataType]] | None = None,
    ) -> pl.DataFrame:
        """Apply functions to subtomograms.

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
        all_tasks: list[DaskTaskList[_R]] = []
        if more_funcs:
            func = [func] + list(more_funcs)  # type: ignore
        if _is_iterable_of_funcs(func):
            funcs = list(func)
        else:
            funcs: list[AggFunction] = [func]  # type: ignore
        if schema is None:
            schema = [fn.__name__ for fn in funcs]
        if isinstance(schema, list):
            if len(set(schema)) != len(schema):
                raise ValueError("Schema names must be unique.")
            elif len(schema) != len(funcs):
                raise ValueError("Schema must have the same length as the functions.")
        elif isinstance(schema, dict):
            if len(schema) != len(funcs):
                raise ValueError("Schema must have the same length as the functions.")
        else:
            raise TypeError("Schema must be a list or a dict.")
        if isinstance(self.output_shape, Unset):
            raise ValueError("Output shape is unknown.")
        for fn in funcs:
            tasks = self.construct_mapping_tasks(fn, output_shape=self.output_shape)
            all_tasks.append(tasks)
        all_results = compute(all_tasks)
        df_input = [np.array(r) for r in all_results]
        return pl.DataFrame(df_input, schema=schema)

    def fsc(
        self,
        mask: TemplateInputType | None = None,
        seed: int | None = 0,
        n_set: int = 1,
        dfreq: float = 0.05,
    ) -> pl.DataFrame:
        """Calculate Fourier shell correlation.

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
        zero_norm: bool = True,
    ) -> tuple[pl.DataFrame, NDArray[np.float32]]:
        df, (img0, img1), _ = self.fsc_with_halfmaps(
            mask=mask,
            seed=seed,
            n_set=n_set,
            dfreq=dfreq,
            zero_norm=zero_norm,
            squeeze=False,
        )
        return df, (img0[0] + img1[0]) / 2

    def fsc_with_halfmaps(
        self,
        mask: TemplateInputType | None = None,
        seed: int | None = 0,
        n_set: int = 1,
        dfreq: float | None = None,
        zero_norm: bool = True,
        squeeze: bool = True,
    ) -> FscTuple:
        """Calculate Fourier shell correlation and the resulting half maps.

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
        zero_norm : bool, default is True
            If True, subtract the mean from the half maps to minimize the sharp mask
            edge effect.

        Returns
        -------
        pl.DataFrame and two 3D arrays
            A data frame with FSC results and the average image. Data frame has
            columns "freq", "FSC-0", ..., "FSC-{n-1}" where n is the number of
            split.

        See Also
        --------
        fsc
        """
        if mask is None or isinstance(mask, ImageConverter):
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
        dfq = 1.5 / min(output_shape) if dfreq is None else dfreq
        halves = self.average_split(
            n_set=n_set,
            seed=seed,
            squeeze=False,
            output_shape=output_shape,
        )
        if isinstance(mask, ImageConverter):
            avg = (halves[0, 0] + halves[0, 1]) / 2
            _mask = mask.convert(avg, self.scale)
        if zero_norm:
            halves[:] -= halves.mean()
        fsc_all: dict[str, np.ndarray] = {}
        freq = np.zeros(0, dtype=np.float32)
        for i in range(n_set):
            img0, img1 = halves[i]
            freq, fsc = _utils.fourier_shell_correlation(
                img0 * _mask, img1 * _mask, dfreq=dfq
            )
            fsc_all[f"FSC-{i}"] = fsc

        out: dict[str, NDArray[np.float32]] = {"freq": freq}
        out.update(fsc_all)
        if n_set == 1 and squeeze:
            return FscTuple(pl.DataFrame(out), (halves[0, 0], halves[0, 1]), _mask)
        else:
            return FscTuple(pl.DataFrame(out), (halves[:, 0], halves[:, 1]), _mask)

    def classify(
        self,
        template: TemplateInputType | None = None,
        mask: MaskInputType = None,
        *,
        cutoff: float = 0.5,
        n_components: int = 2,
        n_clusters: int = 2,
        seed: int = 0,
        label_name: str = "cluster",
    ) -> ClassificationResult:
        """Classify 3D densities by PCA of wedge-masked differences.

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
        - Heumann, J. M., Hoenger, A., & Mastronarde, D. N. (2011). Clustering and
          variance maps for cryo-electron tomography using wedge-masked differences.
          Journal of structural biology, 175(3), 288-299.
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

        if template is None:
            template = self.average(shape)
        stack = self._prep_classify_stack(template, mask, cutoff, shape)

        if callable(_mask):
            mask_arr = _mask(template)
        else:
            mask_arr = _mask
        clf = PcaClassifier(
            # PCA requires aggregation along the first axis. Rechunk to improve
            # performance.
            stack.rechunk(("auto",) + shape),
            mask_arr,
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
        shapes: set[tuple[int, ...]] = set()
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
        return self.replace(output_shape=shapes.pop())  # type: ignore

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
        return self.replace(molecules=self.molecules.filter(predicate))

    @overload
    def groupby(self, by: IntoExpr) -> LoaderGroup[str, Self]: ...

    @overload
    def groupby(
        self, by: Sequence[IntoExpr] | tuple[IntoExpr, ...]
    ) -> LoaderGroup[tuple[str, ...], Self]: ...

    def groupby(self, by):
        """Group loader by given feature column(s).

        >>> for key, loader in loader.groupby("score"):
        ...     print(key, loader)
        """
        if isinstance(by, (str, pl.Expr)):
            _by = by
        else:
            _by = tuple(by)
        return LoaderGroup._from_loader(self, _by)

    group_by = groupby  # alias

    def _get_output_shape(self, output_shape: _ShapeType) -> tuple[pixel, pixel, pixel]:
        """Normalize the `output_shape` parameter."""
        if output_shape is None:
            if isinstance(self.output_shape, Unset):
                raise ValueError("Output shape is unknown.")
            _output_shape = self.output_shape
        else:
            _output_shape = _misc.normalize_shape(output_shape, ndim=3)
        return _output_shape  # type: ignore

    @overload
    def normalize_template(
        self,
        template: TemplateInputType,
        allow_multiple: Literal[False] = False,
    ) -> NDArray[np.float32]: ...
    @overload
    def normalize_template(
        self,
        template: TemplateInputType,
        allow_multiple: Literal[True],
    ) -> NDArray[np.float32] | list[NDArray[np.float32]]: ...

    def normalize_template(self, template, allow_multiple=False):
        """Resolve any template input type to a 3D array."""
        if isinstance(template, np.ndarray):
            if template.ndim == 3:
                return template
            elif template.ndim == 4 and allow_multiple:
                return list(template)
            else:
                raise ValueError("Template must be a 3D array.")
        elif isinstance(template, ImageProvider):
            return template(self.scale)
        elif isinstance(template, (list, tuple)):
            return [self.normalize_template(t, allow_multiple=False) for t in template]
        raise TypeError(f"Invalid template type: {type(template)}")

    def normalize_mask(self, mask: MaskInputType) -> MaskType:
        """Resolve any mask input type to a 3D array."""
        if isinstance(mask, np.ndarray) or mask is None:
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
    ) -> tuple[None, NDArray[np.float32] | None]: ...

    @overload
    def normalize_input(
        self,
        template: TemplateInputType,
        mask: MaskInputType = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32] | None]: ...

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

    def _prep_align_tasks(
        self,
        model: BaseAlignmentModel,
        max_shifts: nm | tuple[nm, nm, nm],
        backend: Backend,
    ) -> DaskTaskList[AlignmentResult]:
        _max_shifts_px = tuple(np.asarray(max_shifts) / self.scale)
        return self.construct_mapping_tasks(
            model.align,
            max_shifts=_max_shifts_px,
            output_shape=model.input_shape,
            backend=backend,
            var_kwarg=dict(
                quaternion=self.molecules.quaternion(),
                pos=self.molecules.pos / self.scale,
            ),
        )

    def _post_align(
        self,
        results: list[AlignmentResult],
        shape: tuple[int, int, int],
    ) -> Self:
        local_shifts, local_rot, scores = _misc.allocate(len(results))
        for i, result in enumerate(results):
            _, loc_shift, local_rot[i], scores[i] = result
            local_shifts[i] = loc_shift * self.scale

        rotator = Rotation.from_quat(local_rot)
        mole_aligned = self.molecules.linear_transform(local_shifts, rotator)

        mole_aligned.features = self.molecules.features.with_columns(
            _misc.get_feature_list(scores, local_shifts, rotator.as_rotvec()),
        )

        return self.replace(molecules=mole_aligned, output_shape=shape)

    def _prep_align_multi_templates(
        self,
        model: BaseAlignmentModel,
        max_shifts: nm | tuple[nm, nm, nm],
        backend: Backend | None = None,
    ) -> DaskTaskList[AlignmentResult]:
        """Prepare the alignment tasks for multiple templates."""
        _max_shifts_px = np.asarray(max_shifts) / self.scale
        return self.construct_mapping_tasks(
            model.align,
            max_shifts=_max_shifts_px,
            output_shape=model.input_shape,
            backend=backend,
            var_kwarg=dict(
                quaternion=self.molecules.quaternion(),
                pos=self.molecules.pos / self.scale,
            ),
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
            labels[i], loc_shift, local_rot[i], scores[i] = result
            local_shifts[i] = loc_shift * self.scale

        rotator = Rotation.from_quat(local_rot)
        mole_aligned = self.molecules.linear_transform(local_shifts, rotator)

        if remainder > 1:
            labels %= remainder  # type: ignore
        labels = labels.astype(np.uint8)

        feature_list = _misc.get_feature_list(scores, local_shifts, rotator.as_rotvec())
        mole_aligned.features = self.molecules.features.with_columns(
            feature_list + [pl.Series(label_name, labels)]
        )

        return self.replace(molecules=mole_aligned, output_shape=shape)

    def _default_align_kwargs(self) -> dict[str, Any]:
        """Return default keyword arguments for alignment."""
        return {}

    def _update_align_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Update default keyword arguments for alignment."""
        align_kwargs = self._default_align_kwargs()
        align_kwargs.update(kwargs)
        return align_kwargs

    def _prep_classify_stack(
        self,
        template: TemplateInputType,
        mask: MaskInputType,
        cutoff: float = 1.0,
        shape: tuple[int, int, int] | None = None,
    ):
        model = ZNCCAlignment(template, mask, cutoff=cutoff)
        return (
            self.iter_mapping_tasks(
                model.masked_difference,
                output_shape=shape,
                var_kwarg={
                    "quaternion": self.molecules.quaternion(),
                    "pos": self.molecules.pos / self.scale,
                },
            )
            .tolist()
            .tostack(shape=shape, dtype=np.float32)
        )


class ClassificationResult(NamedTuple):
    """Tuple of classification results."""

    loader: LoaderBase
    classifier: PcaClassifier


class FscTuple(NamedTuple):
    """Tuple of FSC results."""

    fsc: pl.DataFrame
    halfmaps: tuple[NDArray[np.float32], NDArray[np.float32]]
    mask: NDArray[np.float32]


def check_input(
    order: int,
    output_shape: pixel | tuple[pixel, pixel, pixel] | Unset,
    scale: float,
    ndim: int,
) -> tuple[int, Unset | tuple[pixel, pixel, pixel], float]:
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

    return order, _output_shape, _scale


def _is_iterable_of_funcs(x: Any) -> TypeGuard[Iterable[AggFunction]]:
    if not hasattr(x, "__iter__"):
        return False
    return all(callable(f) for f in x)
