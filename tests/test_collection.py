from functools import lru_cache
import numpy as np
from acryo import BatchLoader, Molecules
from scipy.spatial.transform import Rotation
import polars as pl


def test_replace():
    collection = BatchLoader()
    assert collection.replace(order=1).order == 1
    assert collection.replace(order=1).corner_safe == collection.corner_safe
    assert collection.replace(output_shape=(1, 2, 3)).output_shape == (1, 2, 3)
    assert collection.replace(scale=0.5).scale == 0.5


@lru_cache(maxsize=1)
def _get_collection():
    collection = BatchLoader()
    img0 = np.zeros((10, 10, 10))
    img1 = np.ones((10, 10, 10))
    collection.add_tomogram(
        img0,
        Molecules(
            np.zeros((4, 3)),
            Rotation.from_quat(np.ones((4, 4))),
        ),
    )
    collection.add_tomogram(
        img1,
        Molecules(
            np.zeros((2, 3)),
            Rotation.from_quat(np.ones((2, 4))),
        ),
    )

    return collection


def test_add_tomograms():
    collection = _get_collection()

    assert list(collection.images.keys()) == [0, 1]
    assert len(collection.molecules) == 6


def test_get_loader():
    collection = _get_collection()
    assert collection.loaders[0].image.mean() == 0
    assert collection.loaders[1].image.mean() == 1


def test_iter_loader():
    collection = _get_collection()
    for _ in collection.loaders:
        pass


def test_filter():
    collection = BatchLoader()
    img0 = np.zeros((10, 10, 10))
    img1 = np.ones((10, 10, 10))
    collection.add_tomogram(
        img0,
        Molecules(
            np.zeros((4, 3)),
            Rotation.from_quat(np.ones((4, 4))),
            features={"a": [0, 0, 0, 1]},
        ),
    )
    collection.add_tomogram(
        img1,
        Molecules(
            np.zeros((2, 3)),
            Rotation.from_quat(np.ones((2, 4))),
            features={"a": [0, 1]},
        ),
    )

    out0 = collection.filter(pl.col("a") == 0)
    out1 = collection.filter(pl.col("a") == 1)
    assert len(out0.molecules) == 4
    assert len(out1.molecules) == 2


def test_averaging():
    collection = _get_collection()
    collection.average((3, 3, 3))


def test_alignment():
    collection = _get_collection()
    out = collection.align(np.ones((3, 3, 3), dtype=np.float32))
    assert len(out.molecules) == 6


def test_align_no_template():
    collection = _get_collection()
    out = collection.replace(output_shape=(3, 3, 3)).align_no_template()
    assert len(out.molecules) == 6
