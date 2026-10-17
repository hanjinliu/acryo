import threading

import numpy as np
from numpy.testing import assert_allclose
import pytest
from dask import array as da, delayed
from dask.array.core import normalize_chunks
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from acryo import Molecules, SubtomogramLoader, BatchLoader
from acryo.alignment import ZNCCAlignment
from acryo._utils import SubvolumeOutOfBoundError, prepare_affine_regions

OUTPUT_SHAPE = (8, 9, 10)


def _random_image_and_molecules(n: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    img = rng.normal(size=(40, 50, 60)).astype(np.float32)
    # some molecules are at the edges so that padding is needed
    pos = np.stack([rng.uniform(-1, s + 0.5, n) for s in img.shape], axis=1)
    rot = Rotation.random(n, random_state=seed)
    return img, Molecules(pos, rot)


class ChunkReader:
    """Build a dask array that reads each chunk lazily and count the reads."""

    def __init__(self, img: np.ndarray, chunks):
        self._img = img
        self._lock = threading.Lock()
        self.nreads = 0
        chunks = normalize_chunks(chunks, shape=img.shape)
        bounds = [np.cumsum((0,) + c) for c in chunks]
        blocks = np.empty(tuple(len(c) for c in chunks), dtype=object)
        for idx in np.ndindex(blocks.shape):
            sl = tuple(slice(b[i], b[i + 1]) for i, b in zip(idx, bounds))
            shape = tuple(s.stop - s.start for s in sl)
            blocks[idx] = da.from_delayed(
                delayed(self._read)(sl), shape=shape, dtype=img.dtype
            )
        self.array = da.block(blocks.tolist())

    def _read(self, sl):
        with self._lock:
            self.nreads += 1
        return self._img[sl].copy()


@pytest.mark.parametrize("chunks", [(40, 50, 60), (13, 17, 19), (7, 50, 11)])
@pytest.mark.parametrize("order", [0, 1, 3])
@pytest.mark.parametrize("corner_safe", [False, True])
def test_chunked_image_gives_same_subtomograms(chunks, order, corner_safe):
    img, mole = _random_image_and_molecules()
    kwargs = dict(order=order, output_shape=OUTPUT_SHAPE, corner_safe=corner_safe)
    ref = SubtomogramLoader(img, mole, **kwargs).asnumpy()
    loader = SubtomogramLoader(da.from_array(img, chunks=chunks), mole, **kwargs)
    assert_allclose(loader.asnumpy(), ref, rtol=1e-6, atol=1e-6)
    each = np.stack([arr.compute() for arr in loader.construct_loading_tasks()])
    assert_allclose(each, ref, rtol=1e-6, atol=1e-6)
    assert_allclose(loader.load([3, 1]), ref[[3, 1]], rtol=1e-6, atol=1e-6)
    assert_allclose(loader.load(2), ref[2], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("order", [0, 1, 3])
def test_float16_image(order):
    # float16 is not supported by scipy.ndimage, so the regions are converted
    img, mole = _random_image_and_molecules()
    img16 = img.astype(np.float16)
    kwargs = dict(order=order, output_shape=OUTPUT_SHAPE)
    ref = SubtomogramLoader(img16.astype(np.float32), mole, **kwargs).asnumpy()
    out = SubtomogramLoader(img16, mole, **kwargs).asnumpy()
    assert out.dtype == np.float32
    assert_allclose(out, ref, rtol=1e-6, atol=1e-6)
    chunked = SubtomogramLoader(
        da.from_array(img16, chunks=(13, 17, 19)), mole, **kwargs
    )
    assert_allclose(chunked.asnumpy(), ref, rtol=1e-6, atol=1e-6)


def test_tomogram_chunks_are_loaded_once():
    img, mole = _random_image_and_molecules(n=60)
    reader = ChunkReader(img, chunks=(13, 17, 19))
    nchunks = int(np.prod(reader.array.numblocks))
    loader = SubtomogramLoader(reader.array, mole, order=1, output_shape=OUTPUT_SHAPE)
    template = img[:8, :9, :10]

    loader.average()
    assert 0 < reader.nreads <= nchunks

    reader.nreads = 0
    loader.align(template, max_shifts=1.0, alignment_model=ZNCCAlignment)
    assert 0 < reader.nreads <= nchunks

    reader.nreads = 0
    loader.construct_landscape(template, max_shifts=1.0).compute()
    assert 0 < reader.nreads <= nchunks


def test_graph_size_does_not_depend_on_molecules():
    img, mole = _random_image_and_molecules(n=100)
    loader = SubtomogramLoader(
        da.from_array(img, chunks=20), mole, order=1, output_shape=OUTPUT_SHAPE
    )
    # the number of graph layers must not grow with the number of molecules
    stack = loader.construct_dask()
    assert len(stack.__dask_graph__().layers) < 10
    task = next(iter(loader.construct_mapping_tasks(np.mean)))
    assert len(task.__dask_graph__().layers) < 10


def test_mapping_results_are_in_molecule_order():
    img, mole = _random_image_and_molecules(n=20)
    loader = SubtomogramLoader(
        da.from_array(img, chunks=20), mole, order=1, output_shape=OUTPUT_SHAPE
    )
    ref = loader.asnumpy()
    out = loader.construct_mapping_tasks(
        lambda x, i, offset: float(x.sum()) + offset + i,
        offset=0.5,
        var_kwarg={"i": np.arange(20)},
    ).compute()
    assert_allclose(out, ref.sum(axis=(1, 2, 3)) + 0.5 + np.arange(20), rtol=1e-5)


def test_extracted_subvolumes(tmp_path):
    img, mole = _random_image_and_molecules(n=10)
    loader = SubtomogramLoader(img, mole, order=1, output_shape=OUTPUT_SHAPE)
    extracted = loader.extract_subtomograms(tmp_path / "subtomograms", chunksize=3)
    ref = loader.asnumpy()
    assert_allclose(extracted.asnumpy(), ref)
    # blocks of the extracted loader contain multiple subvolumes
    assert_allclose(
        extracted.apply(np.mean)["mean"].to_numpy(), ref.mean(axis=(1, 2, 3))
    )


@pytest.mark.parametrize("rotate", [False, True])
def test_extracted_subvolumes_prefiltered(tmp_path, rotate: bool):
    import polars as pl

    img, mole = _random_image_and_molecules(n=6)
    loader = SubtomogramLoader(img, mole, order=3, output_shape=OUTPUT_SHAPE)
    ext_pre = loader.extract_subtomograms(tmp_path / "pre", apply_prefilter=True)
    ext_raw = loader.extract_subtomograms(tmp_path / "raw", apply_prefilter=False)
    ref = loader.asnumpy()
    # stored spline coefficients must not be returned as images
    assert_allclose(ext_pre.asnumpy(), ref, rtol=1e-4, atol=1e-4)
    assert ext_pre.asnumpy().dtype == np.float32

    shifts = np.tile([0.4, -0.3, 0.2], (6, 1))
    rotvec = np.tile([0.1, -0.05, 0.08] if rotate else [0.0, 0.0, 0.0], (6, 1))
    features = [pl.Series(f"align-d{c}", shifts[:, i]) for i, c in enumerate("zyx")]
    features += [pl.Series(f"align-d{c}rot", rotvec[:, i]) for i, c in enumerate("zyx")]
    mole_aligned = mole.spatial_transform(shifts, Rotation.from_rotvec(rotvec))
    mole_aligned.features = ext_pre.molecules.features.with_columns(features)
    # `replace` keeps the prefiltered state, so coefficients are not filtered twice
    aligned_pre = ext_pre.replace(molecules=mole_aligned)
    aligned_raw = ext_raw.replace(molecules=mole_aligned)
    sl = (slice(None),) + tuple(slice(2, s - 2) for s in OUTPUT_SHAPE)
    assert_allclose(aligned_pre.asnumpy()[sl], aligned_raw.asnumpy()[sl], atol=1e-5)
    assert_allclose(
        np.stack(list(aligned_pre.load_iter())), aligned_pre.asnumpy(), atol=1e-6
    )
    # coefficients for order=3 must not be used for order=1
    with pytest.raises(ValueError):
        aligned_pre.replace(order=1)


@pytest.mark.parametrize("apply_prefilter", [False, True])
def test_extracted_subvolumes_pre_transform(tmp_path, apply_prefilter: bool):
    import polars as pl

    img, mole = _random_image_and_molecules(n=6)
    mole = mole.with_features(pl.Series("group", [0, 1, 0, 1, 0, 1]))
    loader = SubtomogramLoader(
        img, mole, order=3, output_shape=OUTPUT_SHAPE, corner_safe=True
    )
    extracted = loader.extract_subtomograms(
        tmp_path / "pre", pre_transform=True, apply_prefilter=apply_prefilter
    )
    # same as cropping from the tomogram because no interpolation is done twice
    assert_allclose(extracted.asnumpy(), loader.asnumpy(), rtol=1e-5, atol=1e-5)

    # subvolumes subject to the alignment results are interpolated only once
    rng = np.random.default_rng(1)
    img_smooth = ndi.gaussian_filter(rng.normal(size=(40, 50, 60)), 1.0)
    img_smooth = img_smooth.astype(np.float32)
    mole_inner = Molecules(
        rng.uniform(15, 25, size=(6, 3)), Rotation.random(6, random_state=1)
    )
    loader_inner = SubtomogramLoader(
        img_smooth,
        mole_inner,
        order=3,
        output_shape=OUTPUT_SHAPE,
        corner_safe=True,
    )
    extracted_inner = loader_inner.extract_subtomograms(
        tmp_path / "inner", pre_transform=True, apply_prefilter=apply_prefilter
    )
    shifts = np.tile([0.4, -0.3, 0.2], (6, 1))
    rotvec = np.tile([0.1, -0.05, 0.08], (6, 1))
    features = [pl.Series(f"align-d{c}", shifts[:, i]) for i, c in enumerate("zyx")]
    features += [pl.Series(f"align-d{c}rot", rotvec[:, i]) for i, c in enumerate("zyx")]
    mole_aligned = mole_inner.spatial_transform(shifts, Rotation.from_rotvec(rotvec))
    mole_aligned.features = extracted_inner.molecules.features.with_columns(features)
    # the ground truth is the interpolation of the whole tomogram, because cropping
    # also suffers from the boundary effect of the spline prefilter
    starts, _, matrices = prepare_affine_regions(
        mole_aligned.pos,
        mole_aligned.rotator,
        img_smooth.shape,
        OUTPUT_SHAPE,
        order=3,
        corner_safe=True,
    )
    coords = np.indices(OUTPUT_SHAPE).reshape(3, -1)
    coords = np.concatenate([coords, np.ones((1, coords.shape[1]))])
    truth = np.stack(
        [
            ndi.map_coordinates(
                img_smooth, (mtx @ coords)[:3] + start[:, np.newaxis], order=3
            ).reshape(OUTPUT_SHAPE)
            for mtx, start in zip(matrices, starts)
        ]
    )
    assert_allclose(
        extracted_inner.replace(molecules=mole_aligned).asnumpy(), truth, atol=1e-3
    )
    # subvolumes and pre-transforms are selected consistently after subsetting
    sub = extracted.filter(pl.col("group") == 1)
    assert_allclose(sub.asnumpy(), loader.asnumpy()[1::2], rtol=1e-5, atol=1e-5)
    assert_allclose(
        sub.head(2).asnumpy(), loader.asnumpy()[[1, 3]], rtol=1e-5, atol=1e-5
    )


def _smooth_image_and_molecules(seed: int = 2):
    rng = np.random.default_rng(seed)
    img = ndi.gaussian_filter(rng.normal(size=(48, 56, 64)), 1.5)
    img = (img / img.std()).astype(np.float32)
    mole = Molecules(
        rng.uniform(22, 26, size=(4, 3)), Rotation.random(4, random_state=seed)
    )
    return img, mole


def test_extracted_subvolumes_align_with_rotation(tmp_path):
    img, mole = _smooth_image_and_molecules()
    shape = (15, 15, 15)
    kwargs = dict(order=1, output_shape=shape, corner_safe=True)
    loader = SubtomogramLoader(img, mole, **kwargs)
    # the template is the first molecule translated and then rotated around x-axis
    rotvec = [0.0, 0.0, np.pi / 2]
    truth = mole[:1].spatial_transform([1.0, -0.8, 0.9], Rotation.from_rotvec(rotvec))
    template = SubtomogramLoader(img, truth, **kwargs).asnumpy()[0]
    extracted = loader.extract_subtomograms(tmp_path / "ext", pre_transform=True)
    aligned = extracted.align(
        template, max_shifts=1.2, rotations=((0, 0), (0, 0), (90, 90))
    )
    drot = aligned.features.select([f"align-d{c}rot" for c in "zyx"]).to_numpy()
    assert_allclose(drot[0], rotvec, atol=1e-4)
    assert_allclose(aligned.molecules.pos[0], truth.pos[0], atol=0.5)
    # subvolumes are the same as those cropped at the aligned molecules
    ref = SubtomogramLoader(img, aligned.molecules, **kwargs).asnumpy()
    sl = (slice(None),) + tuple(slice(2, s - 2) for s in shape)
    assert_allclose(aligned.asnumpy()[sl], ref[sl], atol=1e-4)


def test_extracted_subvolumes_align_twice(tmp_path):
    img, mole = _smooth_image_and_molecules()
    shape = (15, 15, 15)
    kwargs = dict(order=1, output_shape=shape, corner_safe=True)
    loader = SubtomogramLoader(img, mole, **kwargs)
    # the first molecule is translated and rotated around x-axis, and then translated
    # and rotated around z-axis
    rotvec1 = [0.0, 0.0, np.pi / 2]
    rotvec2 = [np.pi / 2, 0.0, 0.0]
    truth1 = mole[:1].spatial_transform([1.0, -0.8, 0.9], Rotation.from_rotvec(rotvec1))
    truth2 = truth1.spatial_transform([-0.6, 0.9, 0.7], Rotation.from_rotvec(rotvec2))
    template1 = SubtomogramLoader(img, truth1, **kwargs).asnumpy()[0]
    template2 = SubtomogramLoader(img, truth2, **kwargs).asnumpy()[0]
    extracted = loader.extract_subtomograms(tmp_path / "ext", pre_transform=True)
    aligned1 = extracted.align(
        template1, max_shifts=1.2, rotations=((0, 0), (0, 0), (90, 90))
    )
    aligned2 = aligned1.align(
        template2, max_shifts=1.2, rotations=((90, 90), (0, 0), (0, 0))
    )
    assert_allclose(aligned2.molecules.pos[0], truth2.pos[0], atol=0.5)
    # "align-d*" features are the results of the latest alignment, the same as the
    # other loaders
    dpos = aligned2.features.select([f"align-d{c}" for c in "zyx"]).to_numpy()
    drot = aligned2.features.select([f"align-d{c}rot" for c in "zyx"]).to_numpy()
    assert_allclose(drot[0], rotvec2, atol=1e-4)
    expected = aligned1.molecules.spatial_transform(dpos, Rotation.from_rotvec(drot))
    assert_allclose(aligned2.molecules.pos, expected.pos, atol=0.01)
    # subvolumes are the same as those cropped at the molecules aligned twice
    ref = SubtomogramLoader(img, aligned2.molecules, **kwargs).asnumpy()
    sl = (slice(None),) + tuple(slice(2, s - 2) for s in shape)
    assert_allclose(aligned2.asnumpy()[sl], ref[sl], atol=1e-4)


def test_extracted_subvolumes_resampled_at_molecules(tmp_path):
    import polars as pl

    img, mole = _smooth_image_and_molecules(seed=3)
    rng = np.random.default_rng(3)
    shifts = rng.uniform(-0.6, 0.6, size=(4, 3))
    rotvec = rng.uniform(-0.15, 0.15, size=(4, 3))
    # "align-d*" features of the molecules to be extracted are kept, but they are not
    # used to transform the subvolumes
    mole = mole.with_features(
        [pl.Series(f"align-d{c}", shifts[:, i]) for i, c in enumerate("zyx")]
    )
    shape = (15, 15, 15)
    kwargs = dict(order=1, output_shape=shape, corner_safe=True)
    loader = SubtomogramLoader(img, mole, **kwargs)
    extracted = loader.extract_subtomograms(tmp_path / "ext", pre_transform=True)
    assert extracted.features.columns[:3] == ["align-dz", "align-dy", "align-dx"]
    assert_allclose(extracted.asnumpy(), loader.asnumpy(), rtol=1e-5, atol=1e-5)

    # subvolumes are resampled at any molecules without "align-d*" features
    sl = (slice(None),) + tuple(slice(2, s - 2) for s in shape)
    moved = extracted.replace(
        molecules=extracted.molecules.drop_features([f"align-d{c}" for c in "zyx"])
    )
    for _ in range(2):
        target = moved.molecules.spatial_transform(shifts, Rotation.from_rotvec(rotvec))
        moved = moved.replace(molecules=target)
        ref = SubtomogramLoader(img, target, **kwargs).asnumpy()
        assert_allclose(moved.asnumpy()[sl], ref[sl], atol=1e-4)
    # subvolumes are selected consistently after subsetting
    sub = moved.replace(molecules=moved.molecules.subset([3, 0, 2]))
    assert_allclose(sub.asnumpy()[sl], ref[[3, 0, 2]][sl], atol=1e-4)


def test_order_optimize():
    img, mole = _random_image_and_molecules(n=40)
    # more than 10 chunks along each axis, so that the chunk labels are not sorted
    # lexicographically
    img_dask = da.from_array(img, chunks=4)
    loader = SubtomogramLoader(img_dask, mole, order=1, output_shape=OUTPUT_SHAPE)
    batch = BatchLoader.from_loaders([loader], order=1, output_shape=OUTPUT_SHAPE)
    for ldr in [loader, batch]:
        optimized = ldr.order_optimize()
        pos = optimized.molecules.pos
        # index of the right-closed chunk interval in (y, x, z) order
        keys = [
            np.searchsorted(np.cumsum([0, *img_dask.chunks[i]]), pos[:, i], side="left")
            for i in [1, 2, 0]
        ]
        assert (np.diff(np.lexsort(keys[::-1])) == 1).all()
        assert_allclose(optimized.order_restore().molecules.pos, mole.pos, atol=1e-5)


def test_batch_loader_construct_dask():
    img, mole = _random_image_and_molecules(n=12)
    loader0 = SubtomogramLoader(img, mole[:5], order=1, output_shape=OUTPUT_SHAPE)
    loader1 = SubtomogramLoader(
        da.from_array(img, chunks=20), mole[5:], order=1, output_shape=OUTPUT_SHAPE
    )
    batch = BatchLoader.from_loaders(
        [loader0, loader1], order=1, output_shape=OUTPUT_SHAPE
    )
    # rotations are slightly changed when molecules are added to a batch loader
    assert_allclose(
        batch.construct_dask().compute(),
        np.concatenate([loader0.asnumpy(), loader1.asnumpy()]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_out_of_bound_error():
    img = np.zeros((20, 20, 20), dtype=np.float32)
    mole = Molecules([[10, 10, 10], [10, 10, 40]])
    loader = SubtomogramLoader(img, mole, output_shape=(6, 6, 6))
    with pytest.raises(SubvolumeOutOfBoundError, match="The 1-th molecule"):
        loader.construct_dask()


def test_empty_molecules():
    img = np.zeros((20, 20, 20), dtype=np.float32)
    loader = SubtomogramLoader(img, Molecules.empty(), output_shape=(6, 6, 6))
    assert loader.construct_dask().shape == (0, 6, 6, 6)
    assert loader.construct_mapping_tasks(np.mean).compute() == []
