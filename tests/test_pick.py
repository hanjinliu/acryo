import numpy as np
from numpy.testing import assert_array_equal
from acryo import pick
from acryo.testing import TomogramGenerator, spiral
import pytest


@pytest.mark.parametrize("scale", [1.0, 0.2, 5.0])
def test_pick_dog(scale: float):
    # sample image
    img = np.zeros((20, 50, 50), dtype=np.float32)
    img[10, 20, 25] = 1
    img[10, 21, 25] = 1
    img[12, 38, 15] = 1
    ans = np.array([[10.0, 20.5, 25.0], [12.0, 38.0, 15.0]]) * scale

    out = pick.DoGPicker(2 * scale, 3.5 * scale).pick_molecules(img, scale)
    assert_array_equal(out.pos, ans)


@pytest.mark.parametrize("scale", [1.0, 0.2, 5.0])
def test_pick_log(scale: float):
    # sample image
    img = np.zeros((20, 50, 50), dtype=np.float32)
    img[10, 20, 25] = 1
    img[10, 21, 25] = 1
    img[12, 38, 15] = 1
    ans = np.array([[10.0, 20.5, 25.0], [12.0, 38.0, 15.0]]) * scale

    out = pick.LoGPicker(3 * scale).pick_molecules(img, scale)
    assert_array_equal(out.pos, ans)


@pytest.mark.parametrize("size", [39, 40, 41])
def test_template_matcher(size: int):
    scale = 0.32
    temp = spiral(shape=(size, size, size))
    # temp.shape == (40, 40, 40)
    gen = TomogramGenerator(temp, grid_shape=(3, 3), noise_sigma=0.1)
    tomo = gen.get_tomogram()

    matcher = pick.ZNCCTemplateMatcher(temp)
    out = matcher.pick_molecules(tomo, scale, min_score=0.5)
    ans = np.array(
        [
            [19.5, 19.5, 19.5],
            [19.5, 19.5, 59.5],
            [19.5, 19.5, 99.5],
            [19.5, 59.5, 19.5],
            [19.5, 59.5, 59.5],
            [19.5, 59.5, 99.5],
            [19.5, 99.5, 19.5],
            [19.5, 99.5, 59.5],
            [19.5, 99.5, 99.5],
        ],
        dtype=np.float32,
    )
    assert out.pos.shape == (9, 3)
    assert_array_equal(out.pos, ans * scale)
