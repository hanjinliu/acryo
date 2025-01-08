import numpy as np
from acryo import TomogramSimulator, Molecules
from acryo.testing import spiral, blobs
import polars as pl


def test_simulator():
    img = spiral()
    sim = TomogramSimulator()
    pos = np.array([[40, 40, 40], [30, 40, 30], [40, 30, 40]])
    vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mole = Molecules.from_rotvec(pos, vec)
    sim.add_molecules(mole, img, name="test")
    sim = sim.replace(scale=0.5, order=1).copy()
    out = sim.simulate((100, 100, 100))
    assert out.shape == (100, 100, 100)


def test_simulator_with_color():
    img = spiral()
    sim = TomogramSimulator()
    pos = np.array([[40, 40, 40], [30, 40, 30], [40, 30, 40]])
    vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mole = Molecules.from_rotvec(pos, vec)
    mole.features = {"value": [0, 0.5, 1]}

    def cmap(df: pl.DataFrame):
        v = df["value"][0]
        return v, 0, 1 - v

    sim.add_molecules(mole, img, name="test")
    sim = sim.replace(scale=0.5, order=1).copy()
    out = sim.simulate((100, 100, 100), colormap=cmap)
    assert out.shape == (3, 100, 100, 100)


def test_simulator_2d():
    img = spiral()
    sim = TomogramSimulator()
    pos = np.array([[40, 40, 40], [30, 40, 30], [40, 30, 40]])
    vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mole = Molecules.from_rotvec(pos, vec)
    sim.add_molecules(mole, img, name="test")
    sim = sim.replace(scale=0.5, order=1).copy()
    out = sim.simulate_2d((100, 100))
    assert out.shape == (100, 100)


def test_simulate_projection():
    img = blobs()
    sim = TomogramSimulator()
    pos = np.array([[40, 40, 40], [30, 40, 30], [40, 30, 40]])
    vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mole = Molecules.from_rotvec(pos, vec)
    sim.add_molecules(mole, img, name="test")
    sim.simulate_projection(
        (100, 100), center=(35, 35, 35), xaxis=(0, 0, 1), yaxis=(0, 1, 0)
    )
    out = sim.simulate_projection(
        (100, 100), center=(35, 35, 35), xaxis=(0, 1, 1), yaxis=(0, -1, 1)
    )
    assert out.shape == (100, 100)


def test_simulate_tilt_series():
    img = blobs()
    sim = TomogramSimulator()
    pos = np.array([[40, 40, 40], [30, 40, 30], [40, 30, 40]])
    vec = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mole = Molecules.from_rotvec(pos, vec)
    sim.add_molecules(mole, img, name="test")
    out = sim.simulate_tilt_series([-30, 0, 45], (80, 80, 80))
    assert out.shape == (3, 80, 80)
