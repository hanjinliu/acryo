from functools import lru_cache
import numpy as np
from acryo import BatchLoader, Molecules, SubtomogramLoader, MockLoader
from acryo.backend import Backend
from acryo.loader import ExtractedSubvolumeLoader
from acryo.tilt import single_axis
from numpy.testing import assert_allclose
import pytest
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation
import polars as pl
from dask import array as da


def test_replace():
    loader = BatchLoader()
    assert loader.replace(order=1).order == 1
    assert loader.replace(output_shape=(1, 2, 3)).output_shape == (1, 2, 3)
    assert loader.replace(scale=0.5).scale == 0.5


@lru_cache(maxsize=1)
def _get_batch_loader():
    loader = BatchLoader()
    rng = np.random.default_rng(42)
    img0 = rng.normal(size=(10, 10, 10))
    img1 = rng.normal(loc=1.0, size=(10, 10, 10))
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


def test_edge_cases():
    rng = np.random.default_rng(0)
    img = da.from_array(rng.normal(size=(10, 10, 10)).astype(np.float32), chunks=5)
    batch = BatchLoader()
    assert list(batch.loaders) == []
    batch.add_tomogram(img, Molecules.empty(), image_id=1)
    assert batch.count() == 0
    # the generated image ID does not conflict
    batch.add_tomogram(img, Molecules(np.full((2, 3), 5.0)))
    assert list(batch.images) == [1, 2]
    with pytest.raises(KeyError):
        batch.loaders[1]  # no molecules
    # sources without molecules are also binned
    template = rng.normal(size=(4, 4, 4)).astype(np.float32)
    batch.add_loader(MockLoader(template, Molecules.empty()), image_id=3)
    binned = batch.binning(2, compute=True)
    assert all(isinstance(img, np.ndarray) for img in binned.images.values())
    assert binned.images[2].shape == (5, 5, 5)
    assert binned._sources[3].output_shape == (2, 2, 2)
    assert binned.binning(1).scale == binned.scale == 2.0
    # the accessor does not keep the loader alive
    accessor = BatchLoader().loaders
    with pytest.raises(RuntimeError):
        len(accessor)


def test_get_loader():
    loader = _get_batch_loader()
    assert loader.loaders[0].molecules.count() == 4
    assert loader.loaders[1].molecules.count() == 2
    BatchLoader.from_loaders(loader.loaders)
    # image IDs are kept if a mapping is given
    rebuilt = BatchLoader.from_loaders(
        {k * 10: sub for k, sub in loader.loaders.iter_loaders()}
    )
    assert list(rebuilt.images) == [0, 10]
    assert rebuilt.features["image-id"].to_list() == [0] * 4 + [10] * 2


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


def test_construct_landscape():
    loader = _get_batch_loader()
    land = loader.construct_landscape(np.ones((3, 3, 3), dtype=np.float32))
    assert land.shape == (6, 3, 3, 3)


def test_binning():
    loader = _get_batch_loader()
    loader.binning(2)


def test_score():
    loader = _get_batch_loader()
    template = np.ones((3, 3, 3), dtype=np.float32)
    template[0, 0, 0] = template[1, 1, 1] = 0.0
    loader.score([template])


def test_add_other_loaders(tmp_path):
    rng = np.random.default_rng(0)
    shape = (9, 9, 9)
    template = rng.normal(size=shape).astype(np.float32)
    mock = MockLoader(template, Molecules(rng.normal(scale=0.5, size=(3, 3))), order=1)
    tomo = SubtomogramLoader(
        rng.normal(size=(20, 20, 20)).astype(np.float32),
        Molecules(rng.uniform(8, 12, size=(4, 3))),
        order=1,
        output_shape=shape,
    )
    extracted = tomo.extract_subtomograms(tmp_path / "extracted")
    batch = BatchLoader.from_loaders([tomo, mock, extracted], order=1)
    assert list(batch.images) == [0]
    assert isinstance(batch.loaders[1], MockLoader)
    assert isinstance(batch.loaders[2], ExtractedSubvolumeLoader)
    refs = [tomo.asnumpy(), mock.asnumpy(), extracted.asnumpy()]
    assert_allclose(batch.asnumpy(shape), np.concatenate(refs), atol=1e-5)
    assert batch.filter(pl.col("image-id") == 1).count() == 3
    aligned = batch.replace(output_shape=shape).align(template, max_shifts=1.0)
    assert aligned.molecules.count() == 11
    with pytest.raises(TypeError):
        batch.add_loader(object())  # type: ignore

    # each loader is binned by its own method
    binned = batch.binning(3)
    assert binned.scale == 3.0
    assert binned.images[0].shape == (6, 6, 6)
    subs = dict(binned.loaders.iter_loaders())
    assert subs[1].output_shape == subs[2].output_shape == (3, 3, 3)
    # molecules of the mock loader are relative to the true center
    assert_allclose(subs[1].molecules.pos, mock.molecules.pos, atol=1e-6)
    assert_allclose(subs[0].molecules.pos, tomo.molecules.pos - 1.0, atol=1e-6)
    assert_allclose(subs[2].molecules.pos, extracted.molecules.pos - 1.0, atol=1e-6)
    refs = [tomo.binning(3).asnumpy((3, 3, 3)), mock.binning(3).asnumpy()]
    refs.append(extracted.binning(3).asnumpy())
    assert_allclose(binned.asnumpy((3, 3, 3)), np.concatenate(refs), atol=1e-5)


def test_batch_of_extracted_loaders(tmp_path):
    # extracted subvolumes of a tomogram are joined by a batch loader
    rng = np.random.default_rng(2)
    loader = SubtomogramLoader(
        ndi.gaussian_filter(rng.normal(size=(24, 24, 24)), 2).astype(np.float32),
        Molecules(rng.uniform(4, 8, size=(8, 3)), Rotation.random(8, random_state=2)),
        order=1,
        scale=0.5,
        output_shape=(7, 7, 7),
    )
    extracted = [
        loader.replace(molecules=loader.molecules[sl]).extract_subtomograms(
            tmp_path / f"{i}", apply_prefilter=True
        )
        for i, sl in enumerate([slice(0, 3), slice(3, 8)])
    ]
    first = extracted[0]
    batch = BatchLoader.from_loaders(
        extracted, order=first.order, scale=first.scale, output_shape=first.output_shape
    )
    assert batch.molecules.count() == 8
    assert batch.output_shape == loader.output_shape
    assert_allclose(batch.asnumpy(), loader.asnumpy(), atol=1e-4)
    assert_allclose(batch.average(), loader.average(), atol=1e-4)
    fsc0, avg0 = batch.fsc_with_average(seed=0)
    fsc1, avg1 = loader.fsc_with_average(seed=0)
    assert_allclose(avg0, avg1, atol=1e-4)
    assert_allclose(fsc0["FSC-0"], fsc1["FSC-0"], atol=1e-4)
    # the output shape of the extracted subvolumes is also binned
    binned = batch.binning(2).replace(output_shape=(3, 3, 3))
    assert binned.scale == 1.0
    refs = [ext.binning(2).asnumpy() for ext in extracted]
    assert_allclose(binned.asnumpy(), np.concatenate(refs))


def test_interleaved_molecules():
    # molecules of the images are not contiguous
    rng = np.random.default_rng(1)
    shape = (5, 5, 5)
    tilt_models = [single_axis((-60, 60)), None, single_axis((-40, 50))]
    loaders = [
        SubtomogramLoader(
            rng.normal(loc=i, size=(20, 20, 20)).astype(np.float32),
            Molecules(
                rng.uniform(8, 12, size=(4, 3)), Rotation.random(4, random_state=i)
            ),
            order=1,
            output_shape=shape,
            tilt_model=tilt_models[i],
        )
        for i in range(3)
    ]
    batch = BatchLoader.from_loaders(loaders, order=1, output_shape=shape)
    order = [5, 0, 9, 3, 10, 1, 4, 11, 2, 8, 6, 7]
    shuffled = batch.replace(molecules=batch.molecules.subset(order))
    image_ids = [1, 0, 2, 0, 2, 0, 1, 2, 0, 2, 1, 1]
    assert shuffled.features["image-id"].to_list() == image_ids

    ref = batch.asnumpy()[order]
    assert len(batch.construct_loading_tasks()) == len(order)
    assert_allclose(shuffled.asnumpy(), ref, atol=1e-6)
    each = np.stack([task.compute() for task in shuffled.construct_loading_tasks()])
    assert_allclose(each, ref, atol=1e-6)
    assert_allclose(shuffled.load(3), ref[3], atol=1e-6)
    ft = batch.construct_wedge_weighted_ft(shape, Backend()).compute()
    assert_allclose(
        shuffled.construct_wedge_weighted_ft(shape, Backend()).compute(),
        ft[order],
        atol=1e-5,
    )

    template = rng.normal(size=shape).astype(np.float32)
    aligned = batch.align(template, max_shifts=1.0)
    aligned_shuffled = shuffled.align(template, max_shifts=1.0)
    assert_allclose(aligned_shuffled.molecules.pos, aligned.molecules.pos[order])
    assert aligned_shuffled.features.equals(aligned.features[order])
    templates = [template, rng.normal(size=shape).astype(np.float32)]
    aligned = batch.align(templates, max_shifts=1.0)
    aligned_shuffled = shuffled.align(templates, max_shifts=1.0)
    assert aligned_shuffled.features.equals(aligned.features[order])
    assert_allclose(
        shuffled.score(templates), batch.score(templates)[:, order], atol=1e-6
    )
    land = batch.construct_landscape(template).compute()
    assert_allclose(shuffled.construct_landscape(template).compute(), land[order])
