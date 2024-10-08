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


def test_align_no_template():
    skip_if_cupy()
    loader = _get_mock_loader()
    out = loader.align_no_template()
    assert len(out.molecules) == 4
    loader = _get_mock_loader(degrees=[-45, 0, 45], noise=0.01)
    out = loader.align_no_template()
    assert len(out.molecules) == 4
