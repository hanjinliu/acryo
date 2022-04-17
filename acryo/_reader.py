from __future__ import annotations
import os
from typing import Any, NamedTuple, Callable
import numpy as np
from dask import array as da


class ImageData(NamedTuple):
    image: da.core.Array
    scale: float


Loader = Callable[[str], tuple[np.memmap, float]]


class ImageReader:
    def __init__(self):
        self._reader: dict[str, Loader] = {}

    def register(self, *ext: str):
        ext_list: list[str] = []
        for _ext in ext:
            if not _ext.startswith("."):
                _ext = "." + _ext
            ext_list.append(_ext)

        def _register(f: Loader):
            nonlocal ext_list
            for ext in ext_list:
                self._reader[ext] = f

        return _register

    def imread(self, path: str, chunks: Any = "auto") -> ImageData:
        _, ext = os.path.splitext(path)
        img, scale = self._reader[ext](path)
        return ImageData(image=as_dask(img, chunks=chunks), scale=scale)


READER = ImageReader()


def imread(path: str, chunks: Any = "auto") -> ImageData:
    return READER.imread(path, chunks=chunks)


@READER.register(".tif", ".tiff")
def open_tif(path: str):
    from tifffile import TiffFile

    with TiffFile(path) as tif:
        pagetag = tif.series[0].pages[0].tags  # type: ignore

        tags = {v.name: v.value for v in pagetag.values()}
        scale = tags["XResolution"][1]

        img = tif.asarray(out="memmap")

    return img, scale


@READER.register(".mrc", ".rec", ".map")
def open_mrc(path: str):
    import mrcfile

    with mrcfile.mmap(path, mode="r") as mrc:
        scale = mrc.voxel_size.x
        img = mrc.data

    return img, scale


def as_dask(mmap: np.memmap, chunks: Any = "auto") -> da.core.Array:
    img = da.from_array(mmap, chunks=chunks, meta=np.array([])).map_blocks(
        np.asarray, dtype=mmap.dtype
    )
    return img
