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
)
from typing_extensions import ParamSpec, Self, Concatenate

if TYPE_CHECKING:
    from dask import array as da
    from dask.delayed import Delayed

_P = ParamSpec("_P")
_R = TypeVar("_R")


class DaskTaskPool(Generic[_P, _R]):
    def __init__(self, func: Delayed) -> None:
        self._func = func
        self._tasks = []

    @classmethod
    def from_func(cls, func: Callable[_P, _R]) -> Self:
        from dask import delayed

        return cls(delayed(func))

    def count(self) -> int:
        return len(self._tasks)

    def add_task(self, *args: _P.args, **kwargs: _P.kwargs) -> Self:
        self._tasks.append(self._func(*args, **kwargs))
        return self

    def compute(self) -> list[_R]:
        from dask import compute

        return compute(self._tasks)[0]

    def tolist(self, shape: tuple[int, ...], dtype: Any) -> DaskArrayList:
        from dask import array as da

        return DaskArrayList(
            da.from_delayed(task, shape=shape, dtype=dtype) for task in self._tasks
        )

    def as_stack(self, shape: tuple[int, ...], dtype: Any, axis: int = 0) -> da.Array:
        from dask import array as da

        return da.stack(self.tolist(shape, dtype), axis=axis)

    def __iter__(self) -> Iterator[Delayed]:
        return iter(self._tasks)


class DaskArrayList(Sequence["da.Array"]):
    def __init__(self, arrays: Iterable[da.Array]):
        self._arrays = list(arrays)

    def __len__(self) -> int:
        return len(self._arrays)

    def __iter__(self) -> Iterator[da.Array]:
        return iter(self._arrays)

    def __getitem__(self, index: int) -> da.Array:
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
