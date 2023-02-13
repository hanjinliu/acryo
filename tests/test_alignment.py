import numpy as np
from numpy.testing import assert_allclose
from acryo import SubtomogramLoader
from acryo.alignment import (
    ZNCCAlignment,
    PCCAlignment,
    BaseAlignmentModel,
    rotate,
    euler_to_quat,
)
from acryo.testing import TomogramGenerator, spiral
from scipy import ndimage as ndi
import pytest


@pytest.mark.parametrize("alignment_model", [ZNCCAlignment, PCCAlignment])
@pytest.mark.parametrize("rotations", [None, ((25, 25), (25, 25), (25, 25))])
def test_run(alignment_model: type[BaseAlignmentModel], rotations):
    scale = 0.32
    temp = spiral()
    gen = TomogramGenerator(
        temp, grid_shape=(3, 3), noise_sigma=0.1, rotations=rotations
    )
    tomo = gen.get_tomogram()
    mole = gen.sample_molecules(max_distance=1.0, scale=scale)
    loader = SubtomogramLoader(tomo, mole, order=0, scale=scale)
    out = loader.align(
        template=temp,
        max_shifts=1.2,
        alignment_model=alignment_model,
        rotations=rotations,
        tilt_range=(-60, 60),
    )
    ave = out.average()
    coef = np.corrcoef(ave.ravel(), temp.ravel())
    assert coef[0, 1] > 0.75  # check results are well aligned


scale = 0.32
temp = spiral()
gen = TomogramGenerator(temp, grid_shape=(3, 3), noise_sigma=0.5)
tomo = gen.get_tomogram()
mole = gen.sample_molecules(max_distance=0.1, scale=scale)


def test_fsc():
    loader = SubtomogramLoader(
        tomo, mole, order=0, scale=scale, output_shape=temp.shape
    )
    loader.fsc()
    loader.fsc(mask=(temp > np.mean(temp)).astype(np.float32))


@pytest.mark.parametrize("shift", [[1, 2, 2], [-4, 3, 2]])
@pytest.mark.parametrize("rot", [[15, 0, 15], [-15, 15, 15], [0, 0, -15]])
def test_fit(shift, rot):
    rotations = ((15, 15), (15, 15), (15, 15))
    model = ZNCCAlignment(temp, rotations=rotations)
    temp_transformed = temp * 4 + np.mean(temp)  # linear transformation to input image
    img = ndi.shift(rotate(temp_transformed, rot, cval=np.min), shift=shift)
    imgout, result = model.fit(img, (5, 5, 5))
    assert_allclose(result.quat, euler_to_quat(rot))
    assert_allclose(result.shift, shift)
    coef = np.corrcoef(imgout.ravel(), temp.ravel())
    assert coef[0, 1] > 0.95  # check results are well aligned


def test_pca_classify():
    loader = SubtomogramLoader(
        tomo, mole, order=0, scale=scale, output_shape=temp.shape
    )
    mask = temp > np.mean(temp)
    loader.classify(mask.astype(np.float32), tilt_range=(-60, 60))
