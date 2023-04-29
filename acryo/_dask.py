# pyright: reportPrivateImportUsage=false
from __future__ import annotations
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    TypeVar,
    TYPE_CHECKING,
    Iterator,
    Sequence,
    overload,
)
from typing_extensions import ParamSpec, Self, Concatenate

if TYPE_CHECKING:
    from dask import array as da
    from dask.delayed import Delayed

_P = ParamSpec("_P")
_R = TypeVar("_R")


class DaskTaskIterator(Generic[_R]):
    def __init__(self, tasks: Iterable[da.Array | Delayed]) -> None:
        self._iter = tasks

    def __iter__(self) -> Iterator[da.Array | Delayed]:
        return iter(self._iter)

    def tolist(self) -> DaskTaskList[_R]:
        return DaskTaskList(self._iter)


class DaskTaskList(Generic[_R]):
    def __init__(self, tasks: Iterable[da.Array | Delayed]) -> None:
        self._tasks = list(tasks)

    def count(self) -> int:
        return len(self._tasks)

    def compute(self) -> list[_R]:
        from dask import compute

        return compute(self._tasks)[0]

    def asarrays(self, shape: tuple[int, ...], dtype: Any) -> DaskArrayList:
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


class DaskArrayList(Sequence["da.Array"]):
    def __init__(self, arrays: Iterable[da.Array]):
        self._arrays = list(arrays)

    def __len__(self) -> int:
        return len(self._arrays)

    def __iter__(self) -> Iterator[da.Array]:
        return iter(self._arrays)

    @overload
    def __getitem__(self, index: int) -> da.Array:
        ...  # fmt: skip

    @overload
    def __getitem__(self, index: slice) -> list[da.Array]:
        ...  # fmt: skip

    def __getitem__(self, index):
        return self._arrays[index]

    @classmethod
    def concat(cls, obj: Iterable[Iterable[da.Array]]) -> Self:
        import itertools

        return cls(itertools.chain(*obj))

    def compute(self) -> list[da.Array]:
        from dask import compute

        return compute(self)[0]

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

    def enumerate(self) -> Iterator[tuple[int, da.Array]]:
        return enumerate(self)
