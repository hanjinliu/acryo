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


def test_basic_calculation():
    @provider_function
    def f0(scale: float, value=0):
        return np.full((2, 2, 2), value)

    assert np.all((f0(1) + f0(2))(0.1) == 3)
    assert np.all((f0(1) - f0(2))(0.1) == -1)
    assert np.all((f0(1) * f0(2))(0.1) == 2)
    assert np.all((f0(1) / f0(2))(0.1) == 0.5)
