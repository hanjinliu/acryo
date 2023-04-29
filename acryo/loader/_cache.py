# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import NamedTuple, Mapping
import tempfile
from contextlib import contextmanager
import numpy as np
from dask import array as da


class _Subtomogram(NamedTuple):
    array: da.Array


class SubtomogramCache(Mapping[int, _Subtomogram]):
    def __init__(self, dir=None) -> None:
        self._dict: dict[int, _Subtomogram] = {}
        self._cache_dir = dir

    def __getitem__(self, id_: int) -> _Subtomogram:
        return self._dict[id_]

    def __len__(self) -> int:
        return len(self._dict)

    def __iter__(self):
        return iter(self._dict)

    def cache_array(self, dsk: da.Array, id_) -> da.Array:
        shape = dsk.shape
        with tempfile.NamedTemporaryFile(dir=self._cache_dir) as ntf:
            mmap = np.memmap(ntf, dtype=np.float32, mode="w+", shape=shape)

        da.store(dsk, mmap, compute=True)

        darr = da.from_array(
            mmap,
            chunks=("auto",) + shape[1:],  # type: ignore
            meta=np.array([], dtype=np.float32),
        )
        self._dict[id_] = _Subtomogram(darr)
        return darr

    def get_cache(
        self, id_: int, shape: tuple[int, int, int] | None = None
    ) -> da.Array | None:
        if id_ in self._dict:
            dsk = self._dict[id_].array
            if shape is None or dsk.shape[1:] == shape:
                return dsk
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
