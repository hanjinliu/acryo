from functools import lru_cache
import numpy as np
from acryo import BatchLoader, Molecules, SubtomogramLoader
from scipy.spatial.transform import Rotation
import polars as pl


def test_replace():
    loader = BatchLoader()
    assert loader.replace(order=1).order == 1
    assert loader.replace(order=1).corner_safe == loader.corner_safe
    assert loader.replace(output_shape=(1, 2, 3)).output_shape == (1, 2, 3)
    assert loader.replace(scale=0.5).scale == 0.5


@lru_cache(maxsize=1)
def _get_batch_loader():
    loader = BatchLoader()
    img0 = np.zeros((10, 10, 10))
    img1 = np.ones((10, 10, 10))
    loader.add_tomogram(
        img0,
        Molecules(
            np.zeros((4, 3)),
            Rotation.from_quat(np.ones((4, 4))),
        ),
    )
    loader.add_tomogram(
        img1,
        Molecules(
            np.zeros((2, 3)),
            Rotation.from_quat(np.ones((2, 4))),
        ),
    )
    repr(loader)
    return loader


def test_add_tomograms():
    loader = _get_batch_loader()

    assert list(loader.images.keys()) == [0, 1]
    assert len(loader.molecules) == 6
    assert len(loader.loaders) == 2
    sub = SubtomogramLoader(np.ones((10, 10, 10)), molecules=Molecules.empty())
    loader = loader.copy()
    loader.add_loader(sub)
    loader.add_loader(loader.copy())


def test_get_loader():
    loader = _get_batch_loader()
    assert loader.loaders[0].image.mean() == 0
    assert loader.loaders[1].image.mean() == 1
    BatchLoader.from_loaders(loader.loaders)


def test_iter_loader():
    loader = _get_batch_loader()
    for _ in loader.loaders:
        pass


def test_filter():
    loader = BatchLoader()
    img0 = np.zeros((10, 10, 10))
    img1 = np.ones((10, 10, 10))
    loader.add_tomogram(
        img0,
        Molecules(
            np.zeros((4, 3)),
            Rotation.from_quat(np.ones((4, 4))),
            features={"a": [0, 0, 0, 1]},
        ),
    )
    loader.add_tomogram(
        img1,
        Molecules(
            np.zeros((2, 3)),
            Rotation.from_quat(np.ones((2, 4))),
            features={"a": [0, 1]},
        ),
    )

    out0 = loader.filter(pl.col("a") == 0)
    out1 = loader.filter(pl.col("a") == 1)
    assert len(out0.molecules) == 4
    assert len(out1.molecules) == 2


def test_averaging():
    loader = _get_batch_loader()
    loader.average((3, 3, 3))


def test_alignment():
    loader = _get_batch_loader()
    out = loader.align(np.ones((3, 3, 3), dtype=np.float32))
    assert len(out.molecules) == 6


def test_align_no_template():
    loader = _get_batch_loader()
    out = loader.replace(output_shape=(3, 3, 3)).align_no_template()
    assert len(out.molecules) == 6


def test_binning():
    loader = _get_batch_loader()
    loader.binning(2)
