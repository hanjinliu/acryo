from typing import TYPE_CHECKING
from timeit import default_timer
from contextlib import contextmanager

import numpy as np
from numpy.testing import assert_allclose
import polars as pl
from acryo import SubtomogramLoader, Molecules, pipe
from acryo.alignment import (
    PCCAlignment,
    NCCAlignment,
    ZNCCAlignment,
    BaseAlignmentModel,
)
from acryo._rotation import rotate, euler_to_quat
from acryo.testing import TomogramGenerator, spiral
from acryo._typed_scipy import shift as ndi_shift
import pytest

if TYPE_CHECKING:
    from acryo.alignment._base import TomographyInput


@contextmanager
def measure_time(desc):
    t0 = default_timer()
    yield
    print(f"{desc}: {default_timer() - t0:.3f} s")


@pytest.mark.parametrize("alignment_model", [ZNCCAlignment, NCCAlignment, PCCAlignment])
@pytest.mark.parametrize("rotations", [None, ((25, 25), (25, 25), (25, 25))])
def test_run(alignment_model: type[BaseAlignmentModel], rotations):
    scale = 0.32
    temp = spiral()
    gen = TomogramGenerator(
        temp, grid_shape=(3, 3), noise_sigma=0.1, rotations=rotations
    )
    tomo = gen.get_tomogram()
    mole = gen.sample_molecules(max_distance=1.0, scale=scale)
    loader = SubtomogramLoader(tomo, mole, order=0, scale=scale).copy()
    repr(loader)
    len(loader)
    with measure_time(alignment_model.__name__):
        out = loader.align(
            template=temp,
            max_shifts=1.2,
            alignment_model=alignment_model,
            rotations=rotations,
            tilt=(-60, 60),
        )
    ave = out.average()
    coef = np.corrcoef(ave.ravel(), temp.ravel())
    assert coef[0, 1] > 0.75  # check results are well aligned


scale = 0.32
temp = spiral()
temp_shape: tuple[int, int, int] = temp.shape  # type: ignore
gen = TomogramGenerator(temp, grid_shape=(3, 3), noise_sigma=0.5)
tomo = gen.get_tomogram()
mole = gen.sample_molecules(max_distance=0.1, scale=scale)


def test_fsc():
    loader = SubtomogramLoader(
        tomo, mole, order=0, scale=scale, output_shape=temp_shape
    )
    loader.fsc()
    loader.fsc(mask=(temp > np.mean(temp)).astype(np.float32))


@pytest.mark.parametrize("shift", [[1, 2, 2], [-4, 3, 2]])
@pytest.mark.parametrize("rot", [[15, 0, 15], [-15, 15, 15], [0, 0, -15]])
@pytest.mark.parametrize("alignment_model", [ZNCCAlignment, NCCAlignment, PCCAlignment])
def test_fit(shift, rot, alignment_model: "type[TomographyInput]"):
    rotations = ((15, 15), (15, 15), (15, 15))
    model = alignment_model(temp, rotations=rotations)
    temp_transformed = temp * 4 + np.mean(temp)  # linear transformation to input image
    img = ndi_shift(rotate(temp_transformed, rot, cval=np.min), shift=shift)
    with measure_time(alignment_model.__name__):
        imgout, result = model.fit(img, (5, 5, 5))
    assert_allclose(result.quat, euler_to_quat(rot))
    coef = np.corrcoef(imgout.ravel(), temp.ravel())
    assert coef[0, 1] > 0.95  # check results are well aligned
    assert_allclose(result.shift, shift)


@pytest.mark.parametrize("shift", [[1, 1, 2], [-4, 3, 2]])
@pytest.mark.parametrize("alignment_model", [ZNCCAlignment, NCCAlignment, PCCAlignment])
def test_fit_without_rotation(shift, alignment_model: "type[TomographyInput]"):
    model = alignment_model(temp, rotations=((0, 0), (0, 0), (0, 0)))
    temp_transformed = temp * 4 + np.mean(temp)  # linear transformation to input image
    img = ndi_shift(temp_transformed, shift=shift)
    with measure_time(alignment_model.__name__):
        imgout, result = model.fit(img, (5, 5, 5))
    assert_allclose(result.quat, np.array([0, 0, 0, 1]))
    coef = np.corrcoef(imgout.ravel(), temp.ravel())
    assert coef[0, 1] > 0.95  # check results are well aligned
    assert_allclose(result.shift, shift)


@pytest.mark.parametrize("shift", [[1, 2, 2], [-4, 3, 2]])
@pytest.mark.parametrize("alignment_model", [ZNCCAlignment, NCCAlignment, PCCAlignment])
@pytest.mark.parametrize("upsample", [1, 2])
def test_landscape(shift, alignment_model: "type[TomographyInput]", upsample):
    model = alignment_model(temp)
    temp_transformed = temp * 4 + np.mean(temp)  # linear transformation to input image
    img = ndi_shift(temp_transformed, shift=shift)
    with measure_time(alignment_model.__name__):
        lnd = model.landscape(img, (5, 5, 5), upsample=upsample)
    assert lnd.shape == (10 * upsample + 1,) * 3
    maxima = np.unravel_index(np.argmax(lnd), lnd.shape)
    assert_allclose((np.array(maxima) - 5 * upsample) / upsample, shift)


@pytest.mark.parametrize("shift", [[1.2, 2.4, 4.8], [-4.4, 3.4, 2.2], [0.2, -0.4, 0.6]])
@pytest.mark.parametrize("max_shift", [5.0, 5.3, 5.8])
def test_landscape_float_max_shift(shift, max_shift: float):
    model = ZNCCAlignment(temp)
    upsample = 5
    temp_transformed = temp * 4 + np.mean(temp)  # linear transformation to input image
    img = ndi_shift(temp_transformed, shift=shift)
    with measure_time(ZNCCAlignment.__name__):
        lnd = model.landscape(img, (max_shift, max_shift, max_shift), upsample=upsample)
    maxima = np.unravel_index(np.argmax(lnd), lnd.shape)
    center = (np.array(lnd.shape) - 1) / 2
    assert_allclose((np.array(maxima) - center) / upsample, shift)


def test_with_params():
    model = ZNCCAlignment.with_params(
        rotations=((15, 15), (15, 15), (15, 15)), cutoff=0.4
    )
    assert model.quaternions.shape == (27, 4)
    assert type(model(temp)) is ZNCCAlignment


@pytest.mark.parametrize("alignment_model", [ZNCCAlignment, NCCAlignment, PCCAlignment])
def test_score(alignment_model):
    loader = SubtomogramLoader(
        tomo, mole, order=0, scale=scale, output_shape=temp_shape
    )
    loader.score([temp], alignment_model=alignment_model)
    loader.binning(2).score([temp], alignment_model=alignment_model)


@pytest.mark.parametrize("upsample", [1, 2])
def test_landscape_in_loader(upsample):
    loader = SubtomogramLoader(
        tomo, mole, order=0, scale=scale, output_shape=temp_shape
    )
    mask = temp > np.mean(temp)
    arr: np.ndarray = loader.construct_landscape(
        temp,
        mask=mask.astype(np.float32),
        upsample=upsample,
        max_shifts=[1.0, 1.0, 1.0],
    ).compute()
    assert arr.shape == (
        len(mole),
        6 * upsample + 1,
        6 * upsample + 1,
        6 * upsample + 1,
    )


@pytest.mark.parametrize("upsample", [1, 2])
def test_landscape_in_loader_with_rotation(upsample):
    loader = SubtomogramLoader(
        tomo, mole, order=0, scale=scale, output_shape=temp_shape
    )
    mask = temp > np.mean(temp)
    arr: np.ndarray = loader.construct_landscape(
        temp,
        mask=mask.astype(np.float32),
        upsample=upsample,
        max_shifts=1.0,
        rotations=((15, 15), (15, 15), (15, 15)),
    ).compute()
    assert arr.shape == (
        len(mole),
        27,
        6 * upsample + 1,
        6 * upsample + 1,
        6 * upsample + 1,
    )


def test_em_classification():
    loader = SubtomogramLoader(
        tomo, mole, order=0, scale=scale, output_shape=temp_shape
    )
    mask = temp > np.mean(temp)
    avgs = loader.average_split(n_split=2)
    for result in loader.iter_classify_em(avgs, mask.astype(np.float32), max_niter=2):
        assert len(result.class_templates) == 2
        assert result.probs.shape == (len(mole), 2)


def test_load_functions():
    loader = SubtomogramLoader(tomo, mole, order=0, scale=scale, output_shape=(3, 3, 3))
    for subtomo in loader.load_iter():
        assert subtomo.shape == (3, 3, 3)
    assert loader.load(1).shape == (3, 3, 3)
    assert loader.load(slice(2, 4)).shape == (2, 3, 3, 3)
    assert loader.load([1, 4]).shape == (2, 3, 3, 3)


def test_multi_align():
    from acryo import TomogramSimulator, Molecules

    sim = TomogramSimulator(scale=1)
    img0 = np.zeros((9, 9, 9), dtype=np.float32)
    img0[3:5, 3:5, 3:5] = 1
    img1 = np.zeros((9, 9, 9), dtype=np.float32)
    for sl in [
        (4, 4, 4),
        (3, 4, 4),
        (4, 3, 4),
        (4, 4, 3),
        (5, 4, 4),
        (4, 5, 4),
        (4, 4, 5),
    ]:
        img1[sl] = 1

    sim.add_molecules(Molecules([[10, 15, 15], [10, 35, 15]]), img0)
    sim.add_molecules(Molecules([[10, 15, 35]]), img1)

    tomo = sim.simulate((20, 50, 50))
    mole = Molecules([[10, 15, 15], [10, 15, 35], [10, 35, 15]]).translate(
        [[0, 1, 0], [1, 1, 0], [0, 1, -1]]
    )
    loader = SubtomogramLoader(tomo, mole, order=3, scale=1)
    label_name = "labels"
    out = loader.align_multi_templates(
        [img0, img1], max_shifts=2, label_name=label_name
    )
    assert list(out.features[label_name]) == [0, 1, 0]


def test_multi_align_with_single_template():
    from acryo import TomogramSimulator, Molecules

    sim = TomogramSimulator(scale=1)
    img0 = np.zeros((9, 9, 9), dtype=np.float32)
    img0[3:5, 3:5, 3:5] = 1

    sim.add_molecules(Molecules([[10, 15, 15], [10, 35, 15]]), img0)

    tomo = sim.simulate((20, 50, 50))
    mole = Molecules([[10, 15, 15], [10, 15, 35], [10, 35, 15]]).translate(
        [[0, 1, 0], [1, 1, 0], [0, 1, -1]]
    )
    loader = SubtomogramLoader(tomo, mole, order=3, scale=1)
    label_name = "labels"
    out = loader.align_multi_templates([img0], max_shifts=2, label_name=label_name)
    assert list(out.features[label_name]) == [0, 0, 0]


@pytest.mark.parametrize("alignment_model", [ZNCCAlignment, NCCAlignment, PCCAlignment])
@pytest.mark.parametrize("lim", [1.2, 0.6, 0.3])
def test_max_shifts(alignment_model: "type[TomographyInput]", lim: float):
    rng = np.random.default_rng(48172)
    tomogram = rng.normal(size=(20, 20, 100))
    template = rng.normal(size=(6, 6, 6)).astype(np.float32)
    scale = 0.4
    pos = np.full((10, 3), 10)
    pos[:, 2] = np.linspace(10, 90, 10)
    mole = Molecules(pos * scale)
    loader = SubtomogramLoader(tomogram, mole, order=1, scale=scale)
    aligned = loader.align(template, max_shifts=lim, alignment_model=alignment_model)
    distances = np.abs(aligned.molecules.pos - mole.pos)
    assert np.all(distances <= lim + 1e-6)


FLOAT_DTYPES = (pl.Float32, pl.Float64)


def test_apply():
    loader = SubtomogramLoader(
        tomo, mole, order=0, scale=scale, output_shape=temp_shape
    )
    df = loader.apply(np.mean, np.std)
    assert df.dtypes[0] in FLOAT_DTYPES
    assert df.dtypes[1] in FLOAT_DTYPES

    assert df.shape == (len(mole), 2)

    df = loader.apply(np.mean, np.std, schema=["mu", "sigma"])
    assert df.dtypes[0] in FLOAT_DTYPES
    assert df.dtypes[1] in FLOAT_DTYPES

    assert df.shape == (len(mole), 2)
    assert df.columns == ["mu", "sigma"]

    df = loader.apply([np.mean, np.std], schema={"mu": pl.Float32, "sigma": pl.Float64})
    assert df.dtypes[0] == pl.Float32
    assert df.dtypes[1] == pl.Float64

    with pytest.raises(ValueError):
        loader.apply([np.mean, np.std], schema=["mu"])
    with pytest.raises(ValueError):
        loader.apply([np.mean, np.std], schema=["mu", "mu"])
    with pytest.raises(ValueError):
        loader.apply([np.mean, np.std], schema={"mu": pl.Float32})

    assert df.shape == (len(mole), 2)
    assert df.columns == ["mu", "sigma"]

    df = loader.apply(np.mean, schema=["mu"])
    assert df.dtypes[0] in FLOAT_DTYPES

    assert df.shape == (len(mole), 1)
    assert df.columns == ["mu"]


def test_reshape():
    loader = SubtomogramLoader(tomo, mole, scale=scale)
    _ones = np.ones((4, 4, 4))
    _atoms = pipe.from_atoms(np.array([[0, 0, 1], [1, 0, 0]]))
    _gaussian = pipe.from_gaussian((4, 4, 4))
    _otsu = pipe.soft_otsu()
    assert loader.reshape(_ones, None).output_shape == (4, 4, 4)
    assert loader.reshape(_ones, None, shape=(4, 4, 4)).output_shape == (4, 4, 4)
    assert loader.reshape(None, _ones).output_shape == (4, 4, 4)
    assert loader.reshape(_gaussian, None).output_shape == (12, 12, 12)
    assert loader.reshape(_gaussian, _otsu).output_shape == (12, 12, 12)

    with pytest.raises(ValueError):
        assert loader.reshape()
    assert loader.reshape(_ones, shape=(5, 5, 5))
    assert loader.reshape(None, _ones, shape=(5, 5, 5))
    assert loader.reshape(_atoms, _ones, shape=(5, 5, 5))


def test_normalize_input():
    loader = SubtomogramLoader(tomo, mole, scale=scale)
    _ones = np.ones((4, 4, 4), dtype=np.float32)
    _atoms = pipe.from_atoms(np.array([[0, 0, 1], [1, 0, 0]]))
    _otsu = pipe.soft_otsu()
    loader.normalize_input(_ones, _otsu)
    loader.normalize_input(None, _ones)
    loader.normalize_input(_atoms, None)
    with pytest.raises(TypeError):
        loader.normalize_input(mask=_otsu)

    # unmatched shape
    t, m = loader.normalize_input(_ones, np.ones((2, 2, 2), dtype=np.float32))
    assert t.shape == (4, 4, 4)
    assert m.shape == (4, 4, 4)
    mask_ans = np.zeros((4, 4, 4), dtype=np.float32)
    mask_ans[1:3, 1:3, 1:3] = 1
    assert_allclose(m, mask_ans)
    t, m = loader.normalize_input(_ones, np.ones((2, 3, 2), dtype=np.float32))
    assert t.shape == (4, 4, 4)
    assert m.shape == (4, 4, 4)
    assert m[0, :, :].sum() == 0
    assert m[-1, :, :].sum() == 0
    assert m[:, :, 0].sum() == 0
    assert m[:, :, -1].sum() == 0
