# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import (
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    TypeVar,
    TYPE_CHECKING,
    Union,
    overload,
)
import numpy as np
from numpy.typing import NDArray
import polars as pl
from dask import array as da

from acryo.alignment import (
    BaseAlignmentModel,
    ZNCCAlignment,
    AlignmentFactory,
    AlignmentResult,
)

from acryo import _utils
from acryo._types import nm
from acryo._dask import compute, DaskTaskList
from acryo.loader import _misc
from acryo.backend import Backend, AnyArray

if TYPE_CHECKING:
    from dask.delayed import Delayed
    from acryo.loader._base import (
        LoaderBase,
        Unset,
        _ShapeType,
        TemplateInputType,
        MaskInputType,
        AggFunction,
    )

IntoExpr = Union[str, pl.Expr]
_K = TypeVar("_K", bound=Union[str, tuple[str, ...]])
_L = TypeVar("_L", bound="LoaderBase")
_R = TypeVar("_R")


class LoaderGroup(Generic[_K, _L]):
    """A groupby-like object for subtomogram loaders."""

    def __init__(
        self,
        it: Iterable[tuple[_K, _L]],
    ):
        self._it = it

    @property
    def keys(self) -> list[_K]:
        """All the keys in the group."""
        return [key for key, _ in self._it]

    @overload
    @classmethod
    def _from_loader(cls, loader: _L, by: IntoExpr) -> LoaderGroup[str, _L]:
        ...

    @overload
    @classmethod
    def _from_loader(
        cls, loader: _L, by: tuple[IntoExpr, ...]
    ) -> LoaderGroup[tuple[str, ...], _L]:
        ...

    @classmethod
    def _from_loader(  # type: ignore[override]
        cls, loader: _L, by: IntoExpr | tuple[IntoExpr, ...]
    ):
        return cls(
            LoaderGroupByIterator(
                loader,
                by,
                order=loader.order,
                scale=loader.scale,
                output_shape=loader.output_shape,
                corner_safe=loader.corner_safe,
            )
        )

    def __iter__(self) -> Iterator[tuple[_K, _L]]:
        yield from self._it

    def average(
        self,
        output_shape: _ShapeType = None,
        *,
        backend: Backend | None = None,
    ) -> ArrayDict[_K]:
        """Calculate average images."""
        xp = backend or Backend()
        tasks = []
        keys: list[_K] = []
        for key, loader in self:
            keys.append(key)
            _output_shape = loader._get_output_shape(output_shape)
            dsk = loader.construct_dask(_output_shape, backend=xp)
            tasks.append(da.mean(dsk, axis=0))

        out: list[AnyArray[np.float32]] = da.compute(tasks)[0]
        return ArrayDict({key: xp.asnumpy(img) for key, img in zip(keys, out)})

    def average_split(
        self,
        n_set: int = 1,
        seed: int | None = 0,
        squeeze: bool = True,
        output_shape: _ShapeType = None,
        *,
        backend: Backend | None = None,
    ) -> ArrayDict[_K]:
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
            If true and n_set is 1, return a dictionary of 4D arrays.
        output_shape : tuple of int, optional
            Output shape of the averaged image. If not given, the default output
            shape of the loader objects will be used.

        Returns
        -------
        dict of np.ndarray
            Averaged images with keys of the group keys.
        """
        xp = backend or Backend()
        rng = np.random.default_rng(seed=seed)

        all_tasks: list[list[da.Array]] = []
        for key, loader in self:
            output_shape = loader._get_output_shape(output_shape)
            dask_array = loader.construct_dask(output_shape=output_shape, backend=xp)
            nmole = dask_array.shape[0]
            tasks: list[da.Array] = []
            for _ in range(n_set):
                ind0, ind1 = _misc.random_splitter(rng, nmole)
                _stack = da.stack(
                    [
                        da.mean(dask_array[ind0], axis=0),
                        da.mean(dask_array[ind1], axis=0),
                    ],
                    axis=0,
                )
                tasks.append(_stack)
            all_tasks.append(tasks)

        computed: list[list[AnyArray[np.float32]]] = da.compute(all_tasks)[0]
        out = ArrayDict()
        for (key, loader), stack in zip(self, computed):
            stack_np = xp.asnumpy(xp.stack(stack, axis=0))
            if squeeze and n_set == 1:
                stack_np: NDArray[np.float32] = stack_np[0]
            out[key] = stack_np
        return out

    def align(
        self,
        template: TemplateInputType | Mapping[_K, TemplateInputType],
        *,
        mask: MaskInputType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] | AlignmentFactory = ZNCCAlignment,
        backend: Backend | None = None,
        **align_kwargs,
    ) -> LoaderGroup[_K, _L]:
        """
        Align subtomograms to the template image.

        This method conduct so called "subtomogram alignment". Only shifts and rotations
        are calculated in this method. To get averaged image, you'll have to run
        "average" method using the resulting LoaderGroup instance.

        Parameters
        ----------
        template : 3D array or ImageProvider or mapping
            Template image.
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
        LoaderGroup
            A loader group instance with updated molecules.
        """
        all_tasks: list[DaskTaskList[AlignmentResult]] = []
        template_map = _normalize_template(template)
        input_shape: tuple[int, int, int] | None = None
        for key, loader in self:
            model = alignment_model(
                loader.normalize_template(template_map[key]),
                loader.normalize_mask(mask),
                **align_kwargs,
            )
            _max_shifts_px = np.asarray(max_shifts) / loader.scale
            tasks = loader.construct_mapping_tasks(
                model.align,
                max_shifts=_max_shifts_px,
                output_shape=model.input_shape,
                backend=backend,
                var_kwarg=dict(
                    quaternion=loader.molecules.quaternion(),
                    pos=loader.molecules.pos / loader.scale,
                ),
            )
            all_tasks.append(tasks)
            input_shape = model.input_shape

        if input_shape is None:
            raise RuntimeError("LoaderGroup has no loader.")

        all_results = compute(all_tasks)
        out: list[tuple[_K, _L]] = []
        for (key, loader), results in zip(self, all_results):
            new = loader._post_align(results, input_shape)
            out.append((key, new))
        return self.__class__(out)

    def align_no_template(
        self,
        *,
        mask: MaskInputType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        output_shape: _ShapeType = None,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        backend: Backend | None = None,
        **align_kwargs,
    ) -> LoaderGroup[_K, _L]:
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
        LoaderGroup
            A loader group with updated molecules.
        """
        xp = backend or Backend()
        avg = self.average(output_shape=output_shape, backend=xp)
        return self.align(
            avg,
            mask=mask,
            max_shifts=max_shifts,
            alignment_model=alignment_model,
            backend=xp,
            **align_kwargs,
        )

    def align_multi_templates(
        self,
        templates: list[TemplateInputType] | Mapping[_K, list[TemplateInputType]],
        *,
        mask: MaskInputType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        label_name: str = "labels",
        **align_kwargs,
    ) -> LoaderGroup[_K, _L]:
        """
        Align subtomograms with multiple template images.

        A multi-template version of :func:`align`. This method calculate cross
        correlation for every template and uses the best local shift, rotation and
        template.

        Parameters
        ----------
        templates: list of 3D arrays or ImageProvider or mapping
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
        LoaderGroup
            A loader group with updated molecules.
        """
        all_tasks: list[DaskTaskList[AlignmentResult]] = []
        template_map = _normalize_template(templates)
        input_shape: tuple[int, int, int] | None = None
        has_rotation = False
        for key, loader in self:
            _tmps = template_map[key]
            model = alignment_model(
                template=[loader.normalize_template(t) for t in _tmps],
                mask=loader.normalize_mask(mask),
                **align_kwargs,
            )
            _max_shifts_px = np.asarray(max_shifts) / loader.scale
            tasks = loader.construct_mapping_tasks(
                model.align,
                max_shifts=_max_shifts_px,
                output_shape=model.input_shape,
                var_kwarg=dict(
                    quaternion=loader.molecules.quaternion(),
                    pos=loader.molecules.pos / loader.scale,
                ),
            )
            all_tasks.append(tasks)
            input_shape = model.input_shape
            has_rotation = model.has_rotation

        if input_shape is None:
            raise RuntimeError("LoaderGroup has no loader.")

        all_results = compute(all_tasks)
        if has_rotation:
            remainder = len(templates)
        else:
            remainder = -1
        out: list[tuple[_K, _L]] = []
        for (key, loader), results in zip(self, all_results):
            new = loader._post_align_multi_templates(
                results,
                input_shape,
                remainder,
                label_name,
            )
            out.append((key, new))
        return self.__class__(out)

    def apply(
        self,
        funcs: AggFunction[_R] | Sequence[AggFunction[_R]],
        schema: list[str] | None = None,
    ) -> DataFrameDict[_K]:
        """
        Apply functions to subtomograms for each group.

        Parameters
        ----------
        funcs : callable or list of callable
            Functions that take a subtomogram as input and return a scalar.
        schema : list[str], optional
            DataFrame schema.

        Returns
        -------
        dict of pl.DataFrame
            Result tables for each group.
        """
        all_tasks: list[list[Delayed]] = []
        if not isinstance(funcs, Sequence):
            _funcs = [funcs]
        else:
            _funcs = funcs
        if schema is None:
            schema = [fn.__name__ for fn in _funcs]
        if len(set(schema)) != len(schema):
            raise ValueError("Schema names must be unique.")
        keys: list[_K] = []
        xp = Backend()
        for key, loader in self:
            output_shape = loader.output_shape
            if not isinstance(output_shape, tuple):
                raise ValueError("LoaderGroup has no output shape.")
            taskset = []
            for fn in _funcs:
                f_as_np = _define_apply_func(fn, xp)
                tasks = loader.construct_mapping_tasks(
                    f_as_np, output_shape=output_shape
                )
                taskset.append(list(tasks))
            all_tasks.append(taskset)
            keys.append(key)
        all_results = da.compute(all_tasks)[0]
        out = DataFrameDict()
        for key, result in zip(keys, all_results):
            out[key] = pl.DataFrame(np.array(result), schema=schema)
        return out

    def fsc(
        self,
        mask: TemplateInputType | None = None,
        seed: int | None = 0,
        n_set: int = 1,
        dfreq: float = 0.05,
    ) -> DataFrameDict[_K]:
        """
        Calculate Fourier shell correlation for each group.

        Parameters
        ----------
        mask : np.ndarray, optional
            Mask image
        seed : random seed, default is 0
            Random seed used to split subtomograms.
        n_set : int, default is 1
            Number of split set of averaged images.
        dfreq : float, default is 0.05
            Frequency sampling width.

        Returns
        -------
        pl.DataFrame
            A data frame with FSC results.
        """
        if mask is None:
            _mask = 1
            output_shape = None
        elif isinstance(mask, np.ndarray):
            _mask = mask
            output_shape = mask.shape
        else:
            _mask = 1
            output_shape = None

        if n_set <= 0:
            raise ValueError("'n_set' must be positive.")

        imgs = self.average_split(
            n_set=n_set,
            seed=seed,
            squeeze=False,
            output_shape=output_shape,
        )
        out = DataFrameDict()
        for key, img in imgs.items():
            fsc_all: dict[str, np.ndarray] = {}
            freq = np.zeros(0, dtype=np.float32)
            for i in range(n_set):
                img0, img1 = img[i]
                freq, fsc = _utils.fourier_shell_correlation(
                    img0 * _mask, img1 * _mask, dfreq=dfreq
                )
                fsc_all[f"FSC-{i}"] = fsc

            fsc_dict: dict[str, NDArray[np.float32]] = {"freq": freq}
            fsc_dict.update(fsc_all)
            out[key] = pl.DataFrame(fsc_dict)
        return out

    def count(self) -> dict[_K, int]:
        """Dictionary of the molecule count in each group."""
        return {key: loader.count() for key, loader in self}

    def filter(
        self,
        predicate: pl.Expr | str | pl.Series | list[bool] | np.ndarray,
    ) -> LoaderGroup[_K, _L]:
        """Filter the molecules in each group."""
        return self.__class__((key, loader.filter(predicate)) for key, loader in self)

    def head(self, n: int = 10) -> LoaderGroup[_K, _L]:
        """Get the first n molecules in each group."""
        return self.__class__((key, loader.head(n)) for key, loader in self)

    def tail(self, n: int = 10) -> LoaderGroup[_K, _L]:
        """Get the last n molecules in each group."""
        return self.__class__((key, loader.tail(n)) for key, loader in self)

    def sample(self, n: int = 10, seed: int | None = None) -> LoaderGroup[_K, _L]:
        """Get n random molecules in each group."""
        return self.__class__((key, loader.sample(n, seed)) for key, loader in self)


class LoaderGroupByIterator(Iterable[tuple[_K, _L]]):
    """Iterator for the loader groupby."""

    def __init__(
        self,
        loader: _L,
        by: IntoExpr | Sequence[IntoExpr],
        order: int,
        scale: float,
        output_shape: tuple[int, int, int] | Unset | None,
        corner_safe: bool,
    ):
        self._loader = loader
        self._by = by
        self._order = order
        self._scale = scale
        self._output_shape = output_shape
        self._corner_safe = corner_safe

    def __iter__(self) -> Iterator[tuple[_K, _L]]:
        loader = self._loader
        index_col_name = ".index"
        if index_col_name in loader.molecules.features:
            index_col_name += "."
        index = pl.Series(index_col_name, np.arange(loader.count()))
        for key, mole in loader.molecules.with_features(index).groupby(self._by):
            _loader = loader.replace(
                molecules=mole.drop_features(index_col_name),
                order=self._order,
                scale=self._scale,
                output_shape=self._output_shape,
                corner_safe=self._corner_safe,
            )
            yield key, _loader  # type: ignore


_T = TypeVar("_T")


def _normalize_template(template: _T | Mapping[_K, _T]) -> Mapping[_K, _T]:
    if isinstance(template, Mapping):
        return template
    from collections import defaultdict

    return defaultdict(lambda: template)


def _define_apply_func(fn, xp: Backend):
    return lambda ar: fn(xp.asnumpy(ar))


class ArrayDict(dict[_K, NDArray[np.float32]]):
    def value_list(self) -> list[NDArray[np.float32]]:
        """Return values as a list."""
        return list(self.values())

    def value_stack(self, axis: int = 0) -> NDArray[np.float32]:
        """Return values as a stack."""
        return np.stack(self.value_list(), axis=axis)


class DataFrameDict(dict[_K, pl.DataFrame]):
    def value_list(self) -> list[pl.DataFrame]:
        """Return values as a list."""
        return list(self.values())

    def value_melt(self, key_label: str = "group") -> pl.DataFrame:
        """Return values as a melted data frame."""
        df_melt = []
        for key, df in self.items():
            df_melt.append(
                df.with_columns(pl.repeat(str(key), pl.len()).alias(key_label))
            )
        return pl.concat(df_melt)
