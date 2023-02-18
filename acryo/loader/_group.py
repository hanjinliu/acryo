from __future__ import annotations

from typing import (
    Generic,
    Hashable,
    Iterable,
    Iterator,
    Callable,
    Any,
    TypeVar,
    TYPE_CHECKING,
)
import numpy as np
from numpy.typing import NDArray
import polars as pl
from dask import array as da

from acryo.alignment import (
    BaseAlignmentModel,
    ZNCCAlignment,
    RotationImplemented,
    AlignmentFactory,
)

from acryo._types import nm
from acryo.loader import _misc

if TYPE_CHECKING:
    from typing_extensions import Self
    from dask.delayed import Delayed
    from numpy.typing import NDArray
    from acryo.loader._base import LoaderBase, _ShapeType
    from acryo.alignment._base import MaskType

_K = TypeVar("_K", bound=Hashable)
_L = TypeVar("_L", bound="LoaderBase")


class LoaderGroup(Generic[_K, _L]):
    """A groupby-like object for subtomogram loaders."""

    def __init__(
        self,
        it: Iterable[tuple[_K, _L]],
    ):
        self._it = it

    @classmethod
    def _from_loader(cls, loader: _L, by: _K) -> Self[_K, _L]:
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
        self, output_shape: tuple[int, ...] | None = None
    ) -> dict[_K, NDArray[np.float32]]:
        """Calculate average image."""
        tasks = []
        keys: list[str] = []
        for key, loader in self:
            keys.append(key)
            if output_shape is None:
                output_shape = loader.output_shape
            dsk = loader.construct_dask(output_shape)
            tasks.append(da.mean(dsk, axis=0))

        out: list[NDArray[np.float32]] = da.compute(tasks)[0]
        return {key: img for key, img in zip(keys, out)}

    def average_split(
        self,
        n_set: int = 1,
        seed: int | None = 0,
        squeeze: bool = True,
        output_shape: _ShapeType = None,
    ) -> dict[_K, NDArray[np.float32]]:
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

        Returns
        -------
        np.ndarray
            Averaged image
        """
        rng = np.random.default_rng(seed=seed)

        all_tasks: list[list[da.Array]] = []
        for key, loader in self:
            output_shape = loader._get_output_shape(output_shape)
            dask_array = loader.construct_dask(output_shape=output_shape)
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

        computed = da.compute(all_tasks)[0]
        out: dict[_K, NDArray[np.float32]] = {}
        for key, loader in self:
            stack = np.stack(computed, axis=0)
            if squeeze and n_set == 1:
                stack = stack[0]
            out[key] = stack
        return out

    def align(
        self,
        template: NDArray[np.float32],
        *,
        mask: MaskType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] | AlignmentFactory = ZNCCAlignment,
        **align_kwargs,
    ) -> Self[_K, _L]:
        model = alignment_model(
            template=template,
            mask=mask,
            **align_kwargs,
        )
        all_tasks: list[list[Delayed]] = []
        for key, loader in self:
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
        all_results = da.compute(all_tasks)[0]
        out: list[tuple[_K, _L]] = []
        for (key, loader), results in zip(self, all_results):
            new = loader._post_align(results, model.input_shape)
            out.append((key, new))
        return self.__class__(out)

    def align_no_template(
        self,
        *,
        mask: MaskType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        output_shape: _ShapeType = None,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Self[_K, _L]:
        out: list[tuple[_K, _L]] = []
        for key, loader in self:
            with loader.cached(output_shape=output_shape):
                template = loader.average(output_shape=output_shape)
                aligned = loader.align(
                    template,
                    mask=mask,
                    max_shifts=max_shifts,
                    alignment_model=alignment_model,
                    **align_kwargs,
                )
            out.append((key, aligned))
        return out

    def align_multi_templates(
        self,
        templates: list[NDArray[np.float32]],
        *,
        mask: MaskType = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        label_name: str = "labels",
        **align_kwargs,
    ) -> Self[_K, _L]:
        model = alignment_model(
            template=list(templates),
            mask=mask,
            **align_kwargs,
        )
        all_tasks: list[list[Delayed]] = []
        for key, loader in self:
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
        all_results = da.compute(all_tasks)[0]
        if isinstance(model, RotationImplemented) and model._n_rotations > 1:
            remainder = len(templates)
        else:
            remainder = -1
        out: list[tuple[_K, _L]] = []
        for (key, loader), results in zip(self, all_results):
            new = loader._post_align_multi_templates(
                results,
                model.input_shape,
                remainder,
                label_name,
            )
            out.append((key, new))
        return self.__class__(out)

    def agg(
        self,
        funcs: list[Callable[[NDArray[np.float32]], Any]],
        schema: list[str] | None = None,
    ) -> pl.DataFrame:
        all_tasks: list[list[Delayed]] = []
        if schema is None:
            schema = [fn.__name__ for fn in funcs]
        for key, loader in self:
            taskset = []
            for fn in funcs:
                tasks = loader.construct_mapping_tasks(
                    fn, output_shape=loader.output_shape
                )
                taskset.append(tasks)
            all_tasks.append(taskset)
        all_results = np.array(da.compute(all_tasks)[0])

        return pl.DataFrame(all_results, schema=schema)

    def count(self) -> dict[_K, int]:
        return {loader.count() for key, loader in self}

    def filter(
        self,
        predicate: pl.Expr | str | pl.Series | list[bool] | np.ndarray,
    ) -> Self[_K, _L]:
        return self.__class__((key, loader.filter(predicate)) for key, loader in self)


class LoaderGroupByIterator:
    """Iterator for the loader groupby."""

    def __init__(
        self,
        loader: _L,
        by: _K,
        order: int,
        scale: float,
        output_shape: tuple[int, ...],
        corner_safe: bool,
    ):
        self._loader = loader
        self._by = by
        self._order = order
        self._scale = scale
        self._output_shape = output_shape
        self._corner_safe = corner_safe

    def __iter__(self) -> Iterator[_K, _L]:
        loader = self._loader
        for key, mole in loader.molecules.groupby(self._by):
            _loader = loader.replace(
                molecules=mole,
                order=self._order,
                scale=self._scale,
                output_shape=self._output_shape,
                corner_safe=self._corner_safe,
            )
            yield key, _loader
