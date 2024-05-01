# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import NamedTuple, Mapping
import tempfile
from contextlib import contextmanager
import numpy as np
from dask import array as da
from acryo.backend import Backend


class _Subtomogram(NamedTuple):
    array: da.Array
    backend: str
    mmap: np.memmap


class SubtomogramCache(Mapping[int, _Subtomogram]):
    """Caching system for subtomogram loading tasks."""

    def __init__(self, dir: str | None = None) -> None:
        self._dict: dict[int, _Subtomogram] = {}
        self._cache_dir = dir

    def __getitem__(self, id_: int) -> _Subtomogram:
        return self._dict[id_]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def cache_array(self, dsk: da.Array, id_: int) -> da.Array:
        shape = dsk.shape
        with tempfile.NamedTemporaryFile(dir=self._cache_dir) as ntf:
            mmap = np.memmap(ntf, dtype=np.float32, mode="w+", shape=shape)

            component = dsk[(0,) * dsk.ndim].compute()
            _is_cupy = type(component).__module__.startswith("cupy")
            if _is_cupy:
                dsk = dsk.map_blocks(lambda x: x.get(), dtype=dsk.dtype)  # type: ignore
            self._imsave(dsk, mmap, ntf)
            darr = da.from_array(
                mmap,
                chunks=("auto",) + shape[1:],  # type: ignore
                meta=np.array([], dtype=np.float32),
            )
            self._dict[id_] = _Subtomogram(darr, "cupy" if _is_cupy else "numpy", mmap)
        return darr

    def get_cache(
        self,
        id_: int,
        shape: tuple[int, int, int] | None,
        backend: Backend | None = None,
    ) -> da.Array | None:
        if id_ in self._dict:
            cache = self._dict[id_]
            dsk = cache.array
            if shape is None or dsk.shape[1:] == shape:
                if backend is None:
                    return dsk
                elif cache.backend == "cupy" and backend.name == "numpy":
                    return dsk.map_blocks(
                        backend.asnumpy, dtype=dsk.dtype
                    )  # type: ignore
                return dsk.map_blocks(backend.asarray, dtype=dsk.dtype)  # type: ignore
        return None

    def delete_cache(self, id_):
        self._dict.pop(id_, None)

    @contextmanager
    def temporal(self):
        old_state = self._dict.copy()
        try:
            yield
        finally:
            self._dict = old_state

    def _imsave(self, dsk: da.Array, mmap: np.memmap, path):
        img_dask = _rechunk_to_ones(dsk)
        chunksize = tuple(int(c) for c in img_dask.chunksize)
        writer = _MemmapArrayWriter(path, mmap.offset, dsk.shape, chunksize)
        da.store(img_dask, writer, compute=True)  # type: ignore


def _rechunk_to_ones(arr: da.Array):
    """Rechunk the array to (1, 1, ..., 1, n, Ny, Nx)"""
    size = np.prod(arr.chunksize)
    shape = arr.shape
    cur_prod = 1
    max_i = arr.ndim
    for i in reversed(range(arr.ndim)):
        cur_prod *= shape[i]
        if cur_prod > size:
            break
        max_i = i
    nslices = max(int(size / np.prod(shape[max_i:])), 1)
    if max_i == 0:
        return arr
    else:
        return arr.rechunk((1,) * (max_i - 1) + (nslices,) + shape[max_i:])


class _MemmapArrayWriter:
    def __init__(
        self,
        path: str,
        offset: int,
        shape: tuple[int, ...],
        chunksize: tuple[int, ...],
    ):
        self._path = path
        self._offset = offset
        self._shape = shape  # original shape
        self._chunksize = chunksize  # chunk size
        # shape = (33, 160, 1000, 1000)
        # chunksize = (1, 16, 1000, 1000)
        border = 0
        for i, c in enumerate(chunksize):
            if c != 1:
                border = i
                break
        self._border = border

    def __setitem__(self, sl: tuple[slice, ...], arr: np.ndarray):
        # efficient: shape = (10, 100, 150) and sl = (3:5, 0:100, 0:150)
        # sl = (0:1, 16:32, 0:1000, 0:1000)

        offset = np.sum([sl[i].start * arr.strides[i] for i in range(self._border + 1)])
        arr_ravel = arr.ravel()
        mmap = np.memmap(
            self._path,
            mode="r+",
            offset=self._offset + offset,
            shape=arr_ravel.shape,
            dtype=arr.dtype,
        )
        mmap[: arr_ravel.size] = arr_ravel
        mmap.flush()
