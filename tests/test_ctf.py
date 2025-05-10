import numpy as np
from acryo.ctf import CTFModel


def test_ctf():
    ctf = CTFModel.from_kv(300, 2.7, defocus=-2.1)
    img = ctf.simulate_image((64, 64), 0.2)
    assert img.shape == (64, 64)
    rng = np.random.default_rng(0)
    img = rng.random((72, 60))
    assert ctf.apply_ctf(img, scale=0.2).shape == (72, 60)
    assert ctf.phase_flip(img, scale=0.2).shape == (72, 60)
