import numpy as np
from acryo import SubtomogramLoader
from acryo.alignment import ZNCCAlignment, PCCAlignment, BaseAlignmentModel
from acryo.testing import TomogramGenerator, spiral
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
    )
    ave = out.average()
    coef = np.corrcoef(ave.ravel(), temp.ravel())
    assert coef[0, 1] > 0.75  # check results are well aligned


def test_fsc():
    scale = 0.32
    temp = spiral()
    gen = TomogramGenerator(temp, grid_shape=(3, 3), noise_sigma=0.5)
    tomo = gen.get_tomogram()
    mole = gen.sample_molecules(max_distance=0.1, scale=scale)
    loader = SubtomogramLoader(
        tomo, mole, order=0, scale=scale, output_shape=temp.shape
    )
    loader.fsc()
    loader.fsc(mask=temp > np.mean(temp))
