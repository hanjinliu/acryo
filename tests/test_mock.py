import numpy as np
from acryo import MockLoader, Molecules, pipe
from acryo.backend import Backend
import pytest


def _get_mock_loader(degrees=None, noise=0.0):
    loader = MockLoader(
        pipe.from_gaussian((5, 5, 5)),
        Molecules(np.zeros((4, 3))),
        degrees=degrees,
        noise=noise,
    )
    return loader


def skip_if_cupy():
    if Backend().name == "cupy":
        pytest.skip(reason="not implemented for cupy yet")


def test_averaging():
    skip_if_cupy()
    loader = _get_mock_loader()
    loader.average()
    loader = _get_mock_loader(degrees=[-45, 0, 45], noise=0.01)
    loader.average()


def test_alignment():
    skip_if_cupy()
    loader = _get_mock_loader()
    out = loader.align(pipe.from_gaussian((5, 5, 5)))
    assert len(out.molecules) == 4
    loader = _get_mock_loader(degrees=[-45, 0, 45], noise=0.01)
    out = loader.align(pipe.from_gaussian((5, 5, 5)))
    assert len(out.molecules) == 4


def test_binning():
    skip_if_cupy()
    from numpy.testing import assert_allclose
    from acryo._utils import bin_image

    rng = np.random.default_rng(0)
    template = rng.normal(size=(8, 8, 8)).astype(np.float32)
    mole = Molecules(np.zeros((2, 3)))
    loader = MockLoader(template, mole, degrees=[-45, 0, 45], central_axis=(1, 0, 0))
    binned = loader.binning(2)
    assert binned.scale == 2.0
    assert binned.output_shape == (4, 4, 4)
    assert_allclose(binned.molecules.pos, mole.pos)
    assert_allclose(binned._central_axis, [1, 0, 0])
    assert_allclose(binned._degrees, [-45, 0, 45])
    assert_allclose(loader.replace(order=1)._central_axis, [1, 0, 0])
    exact = MockLoader(template, mole, order=1).binning(2)
    assert_allclose(exact.asnumpy(), np.stack([bin_image(template, 2)] * 2), atol=1e-5)
    # the template is provided at the binned scale
    provided = _get_mock_loader().binning(2)
    assert provided.scale == 2.0
    assert provided.output_shape == pipe.from_gaussian((5, 5, 5))(2.0).shape
    with pytest.raises(ValueError):
        MockLoader(template[:7], mole).binning(2)
    with pytest.raises(ValueError):
        loader.binning(1.5)
