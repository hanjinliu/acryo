from acryo import _utils
from scipy.spatial.transform import Rotation
import pytest


@pytest.mark.parametrize(
    "shape",
    [
        (50, 60, 70),
        (20, 20, 20),
        (21, 21, 21),
        (70, 70, 70),
        (71, 71, 71),
        (51, 31, 21),
    ],
)
def test_missing_wedge(shape):
    rot = Rotation.from_rotvec([0, 1, 0])
    shape = (50, 60, 70)
    mask = _utils.missing_wedge_mask(rot, (-60, 60), shape)
    assert mask.min() == 0.0
    assert mask.max() == 1.0
    assert mask.shape == shape
