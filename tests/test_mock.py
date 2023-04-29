import numpy as np
from acryo import MockLoader, Molecules, pipe


def _get_mock_loader(degrees=None):
    loader = MockLoader(
        pipe.from_gaussian((5, 5, 5)),
        Molecules(np.zeros((4, 3))),
        degrees=degrees,
    )
    return loader


def test_averaging():
    loader = _get_mock_loader()
    loader.average()


def test_alignment():
    loader = _get_mock_loader()
    out = loader.align(pipe.from_gaussian((5, 5, 5)))
    assert len(out.molecules) == 4


def test_align_no_template():
    loader = _get_mock_loader()
    out = loader.align_no_template()
    assert len(out.molecules) == 4
