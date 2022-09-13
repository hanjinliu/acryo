from acryo import _utils
from scipy.spatial.transform import Rotation


def test_missing_wedge():
    rot = Rotation.from_rotvec([0, 1, 0])
    shape = (50, 60, 70)
    mask = _utils.missing_wedge_mask(rot, (-60, 60), shape)
    assert mask.min() == 0.0
    assert mask.max() == 1.0
    assert mask.shape == shape
