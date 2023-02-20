import numpy as np
from acryo.pipe import provider_function


def test_rescale():
    @provider_function
    def fn(scale: float, value=1.0):
        n = int(10 * scale)
        img = np.full((n, n, n), value)
        return img

    out = fn(2.0)(1.0)
    assert out.shape == (10, 10, 10)
    out = fn(2.0)(2.0)
    assert out.shape == (20, 20, 20)
