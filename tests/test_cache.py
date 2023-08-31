# pyright: reportPrivateImportUsage=false

from functools import lru_cache
import numpy as np
from dask import array as da
import polars as pl
from acryo import SubtomogramLoader, Molecules


@lru_cache(maxsize=1)
def _get_loader():
    loader = SubtomogramLoader(
        da.zeros((10, 10, 10)),
        Molecules(
            pos=np.zeros((4, 3)),
            features={"a": [0, 1, 0, 0]},
        ),
        output_shape=(5, 5, 5),
    )
    return loader


def test_caching():
    loader = _get_loader()
    assert len(loader._CACHE) == 0
    with loader.cached():
        assert len(loader._CACHE) == 1
        assert id(loader) in loader._CACHE
    assert len(loader._CACHE) == 0


def test_cache_filtered_loader():
    loader = _get_loader()
    assert len(loader._CACHE) == 0
    with loader.cached():
        assert len(loader._CACHE) == 1
        loader_filt = loader.filter(pl.col("a") == 0)
        assert len(loader._CACHE) == 2
        assert id(loader) in loader._CACHE
        assert id(loader_filt) in loader._CACHE
    assert len(loader._CACHE) == 0


def test_cache_grouped_loader():
    loader = _get_loader()
    assert len(loader._CACHE) == 0
    with loader.cached():
        assert len(loader._CACHE) == 1
        for key, ldr in loader.groupby("a"):
            assert id(ldr) in loader._CACHE
    assert len(loader._CACHE) == 0
