# pyright: reportPrivateImportUsage=false
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import (
    Any,
    List,
    Callable,
    Generic,
    Iterable,
    SupportsIndex,
    TypeVar,
    TYPE_CHECKING,
    Iterator,
    Sequence,
    MutableSequence,
    overload,
)
from typing_extensions import ParamSpec, Self, Concatenate

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dask import array as da
    from dask.delayed import Delayed

_P = ParamSpec("_P")
_R = TypeVar("_R")
_D = TypeVar("_D", bound=np.generic)


class DaskTaskIterator(Generic[_R]):
    def __init__(self, tasks: Iterable[da.Array | Delayed]) -> None:
        self._iter = tasks

    def __iter__(self) -> Iterator[da.Array | Delayed]:
        return iter(self._iter)

    def tolist(self) -> DaskTaskList[_R]:
        return DaskTaskList(self._iter)


class _DaskComputable(Generic[_R], ABC):
    @abstractmethod
    def _as_dask_list(self) -> list[Any]:
        """Convert to a list that is ready for dask computation"""

    def compute(self) -> list[_R]:
        from dask import compute

        return compute(self._as_dask_list())[0]


class DaskTaskList(_DaskComputable[_R]):
    def __init__(self, tasks: Iterable[da.Array | Delayed]) -> None:
        self._tasks = list(tasks)

    def count(self) -> int:
        return len(self._tasks)

    def _as_dask_list(self) -> list[Any]:
        return self._tasks

    def asarrays(self, shape: tuple[int, ...], dtype: type[_D]) -> DaskArrayList[_D]:
        from dask import array as da

        return DaskArrayList(
            da.from_delayed(task, shape=shape, dtype=dtype) for task in self._tasks
        )

    def tostack(self, shape: tuple[int, ...], dtype: Any, axis: int = 0) -> da.Array:
        from dask import array as da

        return da.stack(self.asarrays(shape, dtype), axis=axis)

    def __iter__(self) -> Iterator[da.Array | Delayed]:
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)


class DaskTaskPool(DaskTaskList[_R], Generic[_P, _R]):
    def __init__(self, func: Delayed) -> None:
        self._func = func
        self._tasks: list[da.Array | Delayed] = []

    @classmethod
    def from_func(cls, func: Callable[_P, _R]) -> Self:
        from dask import delayed

        return cls(delayed(func))

    def add_task(self, *args: _P.args, **kwargs: _P.kwargs) -> Self:
        self._tasks.append(self._func(*args, **kwargs))
        return self

    def add_tasks(self, duplication: int, *args: _P.args, **kwargs: _P.kwargs) -> Self:
        task = self._func(*args, **kwargs)
        self._tasks.extend([task] * duplication)
        return self


class NestedDaskTaskList(
    MutableSequence[_DaskComputable[_R]], _DaskComputable[List[_R]]
):
    def __init__(self, tasks: Iterable[_DaskComputable[_R]]) -> None:
        self._tasks = list(tasks)

    def insert(self, index: int, task: _DaskComputable) -> None:
        return self._tasks.insert(index, task)

    def __getitem__(self, index: int) -> _DaskComputable:
        return self._tasks[index]

    def __setitem__(self, index: int, task: _DaskComputable) -> None:
        self._tasks[index] = task

    def __delitem__(self, index: int) -> None:
        del self._tasks[index]

    def __iter__(self) -> Iterator[_DaskComputable]:
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def _as_dask_list(self) -> list[Any]:
        return [t._as_dask_list() for t in self._tasks]


class DaskArrayList(Sequence["da.Array"], _DaskComputable[NDArray[_D]]):
    def __init__(self, arrays: Iterable[da.Array]):
        self._arrays = list(arrays)

    def __len__(self) -> int:
        return len(self._arrays)

    def __iter__(self) -> Iterator[da.Array]:
        return iter(self._arrays)

    @overload
    def __getitem__(self, index: SupportsIndex) -> da.Array:
        ...

    @overload
    def __getitem__(self, index: slice) -> list[da.Array]:
        ...

    def __getitem__(self, index):
        return self._arrays[index]

    @classmethod
    def concat(cls, obj: Iterable[Iterable[da.Array]]) -> Self:
        import itertools

        return cls(itertools.chain(*obj))

    def _as_dask_list(self) -> list[Any]:
        return self._arrays

    def as_stack(self, axis: int = 0) -> da.Array:
        from dask import array as da

        return da.stack(self, axis=axis)

    def map(
        self,
        func: Callable[Concatenate[da.Array, _P], _R],
        *args: _P.args,
        **kwargs: _P.kwargs,
    ) -> DaskTaskPool[[da.Array], _R]:
        pool = DaskTaskPool.from_func(func)
        [pool.add_task(s, *args, **kwargs) for s in self]
        return pool

    def map_blocks(self, func, *args, **kwargs):
        return DaskArrayList(
            [a.map_blocks(func, *args, **kwargs) for a in self._arrays]
        )

    def enumerate(self) -> Iterator[tuple[int, da.Array]]:
        return enumerate(self)


_R1 = TypeVar("_R1")
_R2 = TypeVar("_R2")


@overload
def compute(arg: _DaskComputable[_R]) -> list[_R]:
    ...


@overload
def compute(
    arg: tuple[_DaskComputable[_R], _DaskComputable[_R1]]
) -> tuple[list[_R], list[_R1]]:
    ...


@overload
def compute(
    arg: tuple[_DaskComputable[_R], _DaskComputable[_R1], _DaskComputable[_R2]]
) -> tuple[list[_R], list[_R1], list[_R2]]:
    ...


@overload
def compute(
    arg: Sequence[_DaskComputable[_R]],
) -> list[list[_R]]:
    ...


def compute(
    arg: _DaskComputable | tuple[_DaskComputable, ...] | Sequence[_DaskComputable]
):
    """Compute dask tasks"""
    from dask import compute

    if isinstance(arg, _DaskComputable):
        return arg.compute()

    elif isinstance(arg, tuple):
        return tuple(compute([a._as_dask_list() for a in arg])[0])

    elif isinstance(arg, list):
        return compute([a._as_dask_list() for a in arg])[0]

    else:
        raise TypeError(f"Invalid type: {type(arg)}")
