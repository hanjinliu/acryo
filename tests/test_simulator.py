import numpy as np
from acryo import TomogramSimulator, Molecules
from acryo.testing import spiral


def test_simulator():
    img = spiral()
    sim = TomogramSimulator()
    pos = np.array([[40, 40, 40], [30, 40, 30], [40, 30, 40]])
    vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mole = Molecules.from_rotvec(pos, vec)
    sim.add_molecules(mole, img, name="test")
    sim = sim.replace(scale=0.5, order=1).copy()
    out = sim.simulate((100, 100, 100))
