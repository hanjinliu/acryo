# pyright: reportPrivateImportUsage=false

from __future__ import annotations
import os
from typing import Any, NamedTuple, Callable, TypeVar
import numpy as np
from dask import array as da


class ImageData(NamedTuple):
    image: da.Array
    scale: float


Loader = Callable[[str], tuple[np.memmap, float]]
_L = TypeVar("_L", bound=Loader)


class ImageReaderRegistry:
    def __init__(self):
        self._reader: dict[str, Loader] = {}

    def register(self, *ext: str) -> Callable[[_L], _L]:
        ext_list: list[str] = []
        for _ext in ext:
            if not _ext.startswith("."):
                _ext = "." + _ext
            ext_list.append(_ext)

        def _register(f: _L) -> _L:
            nonlocal ext_list
            for ext in ext_list:
                self._reader[ext] = f
            return f

        return _register

    def imread(self, path: str, chunks: Any = "auto") -> ImageData:
        _, ext = os.path.splitext(path)
        img, scale = self._reader[ext](path)
        return ImageData(image=as_dask(img, chunks=chunks), scale=scale)

    def imread_array(self, path: str) -> tuple[np.ndarray, float]:
        _, ext = os.path.splitext(path)
        img, scale = self._reader[ext](path)
        return np.asarray(img, dtype=np.float32), scale


REG = ImageReaderRegistry()


def imread(path: str, chunks: Any = "auto") -> ImageData:
    return REG.imread(path, chunks=chunks)


def register(*ext: str) -> Callable[[_L], _L]:
    return REG.register(*ext)


# TODO: check scale unit
@REG.register(".tif", ".tiff")
def open_tif(path: str):
    try:
        from tifffile import TiffFile
    except ImportError as e:
        ext = os.path.splitext(path)[1]
        e.msg = (
            f"No module named tifffile. To read {ext[1:]} files, please\n"
            "$ pip install tifffile"
        )
        raise e

    with TiffFile(path) as tif:
        pagetag = tif.series[0].pages[0].tags  # type: ignore

        tags = {v.name: v.value for v in pagetag.values()}
        scale = tags["XResolution"][1]

        img: np.memmap = tif.asarray(out="memmap")  # type: ignore
    return img, scale


@REG.register(".mrc", ".rec", ".map")
def open_mrc(path: str):
    try:
        import mrcfile
    except ImportError as e:
        ext = os.path.splitext(path)[1]
        e.msg = (
            f"No module named mrcfile. To read {ext[1:]} files, please\n"
            "$ pip install mrcfile"
        )
        raise e

    with mrcfile.mmap(path, mode="r") as mrc:
        scale: float = mrc.voxel_size.x / 10
        img: np.memmap = mrc.data  # type: ignore

    return img, scale


def as_dask(mmap: np.memmap, chunks: Any = "auto") -> da.Array:
    img = da.from_array(mmap, chunks=chunks, meta=np.array([])).map_blocks(
        np.asarray, dtype=mmap.dtype
    )  # type: ignore
    return img
