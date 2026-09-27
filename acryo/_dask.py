# pyright: reportPrivateImportUsage=false
from __future__ import annotations
from abc import ABC, abstractmethod
import itertools
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
    from acryo.backend import Backend

_P = ParamSpec("_P")
_R = TypeVar("_R")
_D = TypeVar("_D", bound=np.generic)

# NOTE: Graphs of subvolumes are built as a single layer of a high level graph, with
# one task per subvolume. Building them by calling ``delayed`` functions with sliced
# dask arrays is much slower, especially in dask>=2025, where each dask collection
# passed to a delayed function is optimized separately and its keys are renamed
# (``ProhibitReuse``). The renaming also makes every subvolume load the tomogram
# chunks by itself, instead of sharing them.


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
            da.from_delayed(task, shape=shape, dtype=dtype)
            for i, task in enumerate(self._tasks)
        )

    def tostack(self, shape: tuple[int, ...], dtype: Any, axis: int = 0) -> da.Array:
        from dask import array as da

        return da.stack(self.asarrays(shape, dtype), axis=axis)

    def __iter__(self) -> Iterator[da.Array | Delayed]:
        return iter(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def extend(self, tasks: Iterable[da.Array | Delayed]) -> None:
        self._tasks.extend(tasks)


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
    def __getitem__(self, index: SupportsIndex) -> da.Array: ...

    @overload
    def __getitem__(self, index: slice) -> list[da.Array]: ...

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
def compute(arg: _DaskComputable[_R]) -> list[_R]: ...


@overload
def compute(
    arg: tuple[_DaskComputable[_R], _DaskComputable[_R1]]
) -> tuple[list[_R], list[_R1]]: ...


@overload
def compute(
    arg: tuple[_DaskComputable[_R], _DaskComputable[_R1], _DaskComputable[_R2]]
) -> tuple[list[_R], list[_R1], list[_R2]]: ...


@overload
def compute(
    arg: Sequence[_DaskComputable[_R]],
) -> list[list[_R]]: ...


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


class SubvolumeRegion:
    """How to assemble a subvolume region from image blocks and crop it."""

    __slots__ = ("index", "grid", "src", "pads", "matrix")

    def __init__(
        self,
        index: int,
        grid: tuple[int, ...],
        src: list[tuple[slice, ...]],
        pads: tuple[tuple[int, int], ...] | None,
        matrix: NDArray[np.float32],
    ):
        self.index = index  # index of the subvolume
        self.grid = grid  # number of blocks along each axis
        self.src = src  # slices of each block (C-order)
        self.pads = pads  # pad width, None if not needed
        self.matrix = matrix  # affine matrix of the assembled (and padded) region


class _SubvolumeCropper:
    """Assemble a subvolume region and crop the subvolume by affine transform."""

    __slots__ = ("output_shape", "order", "backend", "expand")

    def __init__(
        self,
        output_shape: tuple[int, ...],
        order: int,
        backend: Backend,
        expand: bool,
    ):
        self.output_shape = output_shape
        self.order = order
        self.backend = backend
        self.expand = expand

    def __call__(self, blocks: list[Any], region: SubvolumeRegion):
        # only the small pieces of the blocks are copied
        pieces = [block[src] for block, src in zip(blocks, region.src)]
        for axis in reversed(range(len(region.grid))):
            if (n := region.grid[axis]) > 1:
                pieces = [
                    np.concatenate(pieces[k : k + n], axis=axis)
                    for k in range(0, len(pieces), n)
                ]
        xp = self.backend
        subimg = xp.asarray(pieces[0])
        if region.pads is not None:
            subimg = xp.pad(subimg, region.pads, mode="mean")
        out = xp.rotated_crop(
            subimg,
            region.matrix,
            shape=self.output_shape,
            order=self.order,
            cval=xp.mean,
        )
        if self.expand:
            return out[np.newaxis]
        return out


def _iter_subvolume_regions(
    image: da.Array,
    starts: NDArray[np.intp],
    stops: NDArray[np.intp],
    matrices: NDArray[np.float32],
) -> Iterator[tuple[list[tuple[int, ...]], SubvolumeRegion]]:
    """Yield the indices of the image blocks and the region for each subvolume."""
    bounds = [[0, *itertools.accumulate(c)] for c in image.chunks]
    lo = np.maximum(starts, 0)
    hi = np.minimum(stops, image.shape)
    pad_lo = (lo - starts).tolist()
    pad_hi = (stops - hi).tolist()
    need_pad = ((lo != starts) | (hi != stops)).any(axis=1).tolist()
    # indices of the first and the last blocks
    first = np.stack(
        [np.searchsorted(b, lo[:, i], side="right") - 1 for i, b in enumerate(bounds)],
        axis=1,
    ).tolist()
    last = np.stack(
        [
            np.searchsorted(b, hi[:, i] - 1, side="right") - 1
            for i, b in enumerate(bounds)
        ],
        axis=1,
    ).tolist()
    lo_list, hi_list = lo.tolist(), hi.tolist()
    for i in range(starts.shape[0]):
        per_axis: list[list[tuple[int, slice]]] = []
        for b, l, h, b0, b1 in zip(bounds, lo_list[i], hi_list[i], first[i], last[i]):
            per_axis.append(
                [
                    (ib, slice(max(l, b[ib]) - b[ib], min(h, b[ib + 1]) - b[ib]))
                    for ib in range(b0, b1 + 1)
                ]
            )
        block_indices: list[tuple[int, ...]] = []
        src: list[tuple[slice, ...]] = []
        for combo in itertools.product(*per_axis):
            block_indices.append(tuple(c[0] for c in combo))
            src.append(tuple(c[1] for c in combo))
        grid = tuple(len(p) for p in per_axis)
        pads = tuple(zip(pad_lo[i], pad_hi[i])) if need_pad[i] else None
        yield block_indices, SubvolumeRegion(i, grid, src, pads, matrices[i])


def subvolume_stack(
    image: da.Array,
    starts: NDArray[np.intp],
    stops: NDArray[np.intp],
    matrices: NDArray[np.float32],
    output_shape: tuple[int, ...],
    order: int,
    backend: Backend,
) -> da.Array:
    """Construct a stack of subvolumes cropped from the image.

    The i-th subvolume is cropped from the region ``starts[i]:stops[i]`` (padded if
    the region is out of the image) by the affine matrix ``matrices[i]``. Each
    subvolume is a chunk of the returned array. The image blocks are shared between
    subvolumes, that is, each block is loaded only once in a computation.
    """
    from dask import array as da
    from dask.base import tokenize
    from dask.highlevelgraph import HighLevelGraph

    nvol = starts.shape[0]
    if nvol == 0:
        return da.zeros((0,) + tuple(output_shape), dtype=np.float32)
    name = "subvolumes-" + tokenize(
        image.name, starts, stops, matrices, output_shape, order, backend.name
    )
    cropper = _SubvolumeCropper(output_shape, order, backend, expand=True)
    zeros = (0,) * len(output_shape)
    layer = {}
    for block_indices, region in _iter_subvolume_regions(
        image, starts, stops, matrices
    ):
        keys = [(image.name, *idx) for idx in block_indices]
        layer[(name, region.index, *zeros)] = (cropper, keys, region)
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[image])
    chunks = ((1,) * nvol,) + tuple((s,) for s in output_shape)
    meta = np.empty((0,) * (len(output_shape) + 1), dtype=np.float32)
    return da.Array(graph, name, chunks, meta=meta)


def subvolume_array_list(
    image: da.Array,
    starts: NDArray[np.intp],
    stops: NDArray[np.intp],
    matrices: NDArray[np.float32],
    output_shape: tuple[int, ...],
    order: int,
    backend: Backend,
) -> DaskArrayList[np.float32]:
    """Same as ``subvolume_stack`` but each subvolume is an independent array."""
    from dask import array as da
    from dask.base import tokenize
    from dask.highlevelgraph import HighLevelGraph

    prefix = "subvolume-" + tokenize(
        image.name, starts, stops, matrices, output_shape, order, backend.name
    )
    cropper = _SubvolumeCropper(output_shape, order, backend, expand=False)
    zeros = (0,) * len(output_shape)
    chunks = tuple((s,) for s in output_shape)
    meta = np.empty((0,) * len(output_shape), dtype=np.float32)
    arrays: list[da.Array] = []
    for block_indices, region in _iter_subvolume_regions(
        image, starts, stops, matrices
    ):
        name = f"{prefix}-{region.index}"
        keys = [(image.name, *idx) for idx in block_indices]
        layer = {(name, *zeros): (cropper, keys, region)}
        graph = HighLevelGraph.from_collections(name, layer, dependencies=[image])
        arrays.append(da.Array(graph, name, chunks, meta=meta))
    return DaskArrayList(arrays)


class _SubvolumeMapper:
    """Apply a function to the j-th subvolume of a block."""

    __slots__ = ("func", "args", "kwargs", "var_kwarg", "expand")

    def __init__(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        var_kwarg: dict[str, Sequence[Any]],
        expand: bool,
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.var_kwarg = var_kwarg
        self.expand = expand

    def __call__(self, block, j: int, i: int):
        var = {k: v[i] for k, v in self.var_kwarg.items()}
        out = self.func(block[j], *self.args, **self.kwargs, **var)
        if self.expand:
            return out[np.newaxis]
        return out


def _prep_mapping(
    stack: da.Array,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    var_kwarg: dict[str, Iterable[Any]] | None,
    expand: bool,
) -> tuple[da.Array, str, Iterator[tuple[tuple, int]], _SubvolumeMapper]:
    import uuid
    from dask.utils import funcname

    if any(len(c) != 1 for c in stack.chunks[1:]):
        # each block must contain whole subvolumes
        stack = stack.rechunk((stack.chunks[0],) + (-1,) * (stack.ndim - 1))
    nvol = stack.shape[0]
    _var_kwarg: dict[str, Sequence[Any]] = {}
    for k, v in (var_kwarg or {}).items():
        if not (hasattr(v, "__getitem__") and hasattr(v, "__len__")):
            v = list(v)
        if len(v) != nvol:  # type: ignore
            raise ValueError(
                f"Length of variable keyword argument {k!r} is {len(v)}, but the "  # type: ignore
                f"number of subvolumes is {nvol}."
            )
        _var_kwarg[k] = v  # type: ignore
    mapper = _SubvolumeMapper(func, args, kwargs, _var_kwarg, expand)
    name = f"{funcname(func)}-{uuid.uuid4().hex}"
    zeros = (0,) * (stack.ndim - 1)
    block_keys = (
        ((stack.name, ib, *zeros), j)
        for ib, size in enumerate(stack.chunks[0])
        for j in range(size)
    )
    return stack, name, block_keys, mapper


def map_subvolumes(
    stack: da.Array,
    func: Callable[..., _R],
    *args,
    var_kwarg: dict[str, Iterable[Any]] | None = None,
    **kwargs,
) -> DaskTaskList[_R]:
    """Construct delayed tasks of ``func(stack[i], *args, **kwargs, **var_kwarg_i)``.

    All the tasks are in a single layer that refers to the blocks of ``stack``.
    """
    from dask.delayed import Delayed
    from dask.highlevelgraph import HighLevelGraph

    stack, name, block_keys, mapper = _prep_mapping(
        stack, func, args, kwargs, var_kwarg, expand=False
    )
    layer = {(name, i): (mapper, key, j, i) for i, (key, j) in enumerate(block_keys)}
    if len(layer) == 0:
        return DaskTaskList([])
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[stack])
    return DaskTaskList(Delayed(key, graph, layer=name) for key in layer)


def map_subvolumes_to_array(
    stack: da.Array,
    func: Callable[..., Any],
    shape: tuple[int, ...],
    dtype: Any,
    *args,
    var_kwarg: dict[str, Iterable[Any]] | None = None,
    **kwargs,
) -> da.Array:
    """Construct a dask array of ``func(stack[i], *args, **kwargs, **var_kwarg_i)``.

    ``func`` must return an array of the given shape. The returned array has shape
    ``(len(stack), *shape)``.
    """
    from dask import array as da
    from dask.highlevelgraph import HighLevelGraph

    stack, name, block_keys, mapper = _prep_mapping(
        stack, func, args, kwargs, var_kwarg, expand=True
    )
    zeros = (0,) * len(shape)
    layer = {
        (name, i, *zeros): (mapper, key, j, i) for i, (key, j) in enumerate(block_keys)
    }
    nvol = len(layer)
    if nvol == 0:
        return da.zeros((0,) + tuple(shape), dtype=dtype)
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[stack])
    chunks = ((1,) * nvol,) + tuple((s,) for s in shape)
    meta = np.empty((0,) * (len(shape) + 1), dtype=dtype)
    return da.Array(graph, name, chunks, meta=meta)
