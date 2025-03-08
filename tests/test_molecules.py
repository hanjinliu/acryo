from pathlib import Path
import tempfile
from acryo import Molecules
from acryo.molecules import axes_to_rotator
import numpy as np
import polars as pl
from numpy.testing import assert_allclose
import pytest
from scipy.spatial.transform import Rotation

Sq3 = np.sqrt(3)
Sq2 = np.sqrt(2)

values = [
    (
        [1, 0, 0],
        [0, 1 / Sq2, 1 / Sq2],
        [[1, 0, 0], [0, 1 / Sq2, -1 / Sq2], [0, 1 / Sq2, 1 / Sq2]],
    ),
    ([0, 0, -1], [0, 1, 0], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
    (
        [1, 0, 0],
        [0, Sq3 / 2, 1 / 2],
        [[1, 0, 0], [0, Sq3 / 2, -1 / 2], [0, 1 / 2, Sq3 / 2]],
    ),
    (
        [1 / 2, -Sq3 / 2, 0],
        [Sq3 / 2, 1 / 2, 0],
        [[1 / 2, Sq3 / 2, 0], [-Sq3 / 2, 1 / 2, 0], [0, 0, 1]],
    ),
]


@pytest.mark.parametrize("zvec, yvec, mat", values)
def test_matrix(zvec, yvec, mat):
    pos = np.array([0, 0, 0])
    zvec = np.array(zvec)
    yvec = np.array(yvec)
    mat = np.array(mat)[np.newaxis]
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.z[0], zvec, rtol=1e-8, atol=1e-8)
    assert_allclose(mol.y[0], yvec, rtol=1e-8, atol=1e-8)
    out = mol.matrix()
    assert_allclose(out, mat, rtol=1e-6, atol=1e-6)
    assert_allclose(np.cross(mol.y, mol.x, axis=1), mol.z, rtol=1e-8, atol=1e-8)


def test_euler():
    pos = np.array([0, 0, 0])
    zvec = np.array([1, 0, 0])
    yvec = np.array([0, 1 / Sq2, 1 / Sq2])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.euler_angle("ZYX", degrees=True), [[45, 0, 0]])

    pos = np.array([0, 0, 0])
    zvec = np.array([0, 0, -1])
    yvec = np.array([0, 1, 0])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.euler_angle("zyz", degrees=True), [[0, 90, 0]])

    pos = np.array([0, 0, 0])
    zvec = np.array([1 / Sq2, 1 / Sq2, 0])
    yvec = np.array([-1 / Sq2, 1 / Sq2, 0])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.euler_angle("zyx", degrees=True), [[0, 0, 45]])


def test_rotvec():
    pos = np.array([0, 0, 0])
    zvec = np.array([1, 0, 0])
    yvec = np.array([0, 1 / Sq2, 1 / Sq2])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.rotvec(), [[np.pi / 4, 0, 0]])

    pos = np.array([0, 0, 0])
    zvec = np.array([0, 0, -1])
    yvec = np.array([0, 1, 0])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    assert_allclose(mol.rotvec(), [[0, np.pi / 2, 0]])


def test_save_and_load_euler_angle():
    pos = np.array([0, 0, 0])
    zvec = np.array([1, 0.4, 0.1])
    yvec = np.array([0, 1.1, 2])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)
    euler = mol.euler_angle(degrees=True)
    mol2 = Molecules.from_euler(pos, euler, degrees=True)
    assert_allclose(mol2.x, mol.x, rtol=1e-8, atol=1e-8)
    assert_allclose(mol2.y, mol.y, rtol=1e-8, atol=1e-8)
    assert_allclose(mol2.z, mol.z, rtol=1e-8, atol=1e-8)


def test_rotate():
    pos = np.array([0, 0, 0])
    zvec = np.array([1, 0.4, 0.1])
    yvec = np.array([0, 1.1, 2])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)

    rot = Rotation.from_rotvec([0.1, 0.3, -0.2])
    mol2 = mol.rotate_by(rot)

    assert_allclose(rot.apply(mol.z), mol2.z, rtol=1e-8, atol=1e-8)
    assert_allclose(rot.apply(mol.y), mol2.y, rtol=1e-8, atol=1e-8)
    assert_allclose(rot.apply(mol.x), mol2.x, rtol=1e-8, atol=1e-8)

    mol.rotate_by_matrix(rot.as_matrix())
    mol.rotate_by_rotvec(rot.as_rotvec())
    mol.rotate_by_quaternion(rot.as_quat(), copy=False)
    mol.rotate_by_euler_angle(rot.as_euler("zyx"), degrees=True)
    mol.rotate_random(seed=3)


def test_internal_transformation():
    pos = np.array([0, 0, 0])
    zvec = np.array([1, 0, 0])
    yvec = np.array([0, 1 / Sq2, -1 / Sq2])
    mol = Molecules.from_axes(pos, z=zvec, y=yvec)

    # internal translation
    mol0 = mol.translate_internal([2, 0, 0])
    assert_allclose(
        mol0.pos,
        mol.pos + np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
        rtol=1e-8,
        atol=1e-8,
    )
    assert_allclose(mol0.x, mol.x, rtol=1e-8, atol=1e-8)
    assert_allclose(mol0.y, mol.y, rtol=1e-8, atol=1e-8)
    mol1 = mol.translate_internal([1, 1, 1])
    assert_allclose(
        mol1.pos,
        mol.pos + np.array([[1.0, Sq2, 0.0]], dtype=np.float32),
        rtol=1e-8,
        atol=1e-8,
    )
    assert_allclose(mol1.x, mol.x, rtol=1e-8, atol=1e-8)
    assert_allclose(mol1.y, mol.y, rtol=1e-8, atol=1e-8)

    # itnernal rotation
    mol2 = mol.rotate_by_rotvec_internal([-np.pi / 4, 0.0, 0.0])
    assert_allclose(mol2.pos, mol.pos, rtol=1e-8, atol=1e-8)
    assert_allclose(mol2.z, np.array([[1.0, 0.0, 0.0]]), rtol=1e-8, atol=1e-8)
    assert_allclose(mol2.y, np.array([[0.0, 0.0, -1.0]]), rtol=1e-8, atol=1e-8)
    assert_allclose(mol2.x, np.array([[0.0, 1.0, 0.0]]), rtol=1e-8, atol=1e-8)

    mol3 = mol.rotate_by_rotvec_internal([0.0, 0.0, -np.pi / 2])
    assert_allclose(mol3.pos, mol.pos, rtol=1e-8, atol=1e-8)
    assert_allclose(mol3.z, np.array([[0.0, -1 / Sq2, 1 / Sq2]]), rtol=1e-8, atol=1e-8)
    assert_allclose(mol3.y, np.array([[1.0, 0.0, 0.0]]), rtol=1e-8, atol=1e-8)
    assert_allclose(mol3.x, np.array([[0.0, 1 / Sq2, 1 / Sq2]]), rtol=1e-8, atol=1e-8)


def test_features():
    mol = Molecules(
        np.zeros((24, 3)), Rotation.random(24), features={"n": np.arange(24)}
    )
    mol2 = mol.translate([1, 2, 3])
    mol3 = mol[3:17]
    assert_allclose(mol.features, mol2.features)
    assert mol.features is not mol2.features
    assert_allclose(mol3.features, mol.features[3:17])
    mol.translate([1, 2, 3], copy=False)


@pytest.mark.parametrize("sl", [2, slice(2, 5), [2, 4, 6], np.array([2, 4, 6])])
def test_subset(sl):
    mol = Molecules(np.zeros((24, 3)), Rotation.random(24))
    mol.subset(sl)


def test_random_shift():
    mol = Molecules(np.random.random((24, 3)) * 10, Rotation.random(24))
    mol_shifted = mol.translate_random(2.5)
    dvec = mol.pos - mol_shifted.pos  # type: ignore
    dist = np.sqrt(np.sum((dvec) ** 2, axis=1))
    assert np.all(dist <= 2.5)


@pytest.mark.parametrize(
    "feat",
    [
        pl.Series("name", [0, 2, 4, 7]),
        pl.DataFrame({"name0": [0, 1, 3, 4], "name1": [3, 2, 1, 0]}),
        {"name0": [0, 1, 3, 4], "name1": [3, 2, 1, 0]},
    ],
)
def test_setting_features(feat):
    mol = Molecules(np.random.random((4, 3)), Rotation.random(4))
    mol.features = feat


def test_io():
    mol = Molecules(np.random.random((4, 3)), Rotation.random(4))
    mol.features = {"A": [0, 2, 4, 6]}
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        path = root / "test.csv"
        mol.to_csv(path)
        mol.to_file(path)
        mol0 = Molecules.from_csv(path)
        assert_allclose(mol.to_dataframe(), mol0.to_dataframe(), rtol=1e-6, atol=1e-4)
        mol0 = Molecules.from_file(path)
        assert_allclose(mol.to_dataframe(), mol0.to_dataframe(), rtol=1e-6, atol=1e-4)
        path = root / "test.parquet"
        mol.to_parquet(path)
        mol.to_file(path)
        mol0 = Molecules.from_parquet(path)
        assert_allclose(mol.to_dataframe(), mol0.to_dataframe(), rtol=1e-6, atol=1e-4)
        mol0 = Molecules.from_file(path)
        assert_allclose(mol.to_dataframe(), mol0.to_dataframe(), rtol=1e-6, atol=1e-4)


def test_random():
    mol = Molecules.from_random(np.zeros((4, 3)), seed=0)
    assert len(mol) == 4


def test_groupby():
    mol = Molecules(np.zeros((4, 3)), features={"A": [0, 0, 1, 1]})
    grouped = list(mol.groupby("A"))
    assert len(grouped) == 2
    assert grouped[0][0] == 0
    assert grouped[1][0] == 1

    grouped = list(mol.groupby(["A"]))
    assert len(grouped) == 2
    assert grouped[0][0] == (0,)
    assert grouped[1][0] == (1,)


def test_cutby():
    mol = Molecules(np.zeros((4, 3)), features={"A": [0.1, 0.2, 0.3, 0.4]})
    grouped = list(mol.cutby("A", bins=[0.0, 0.2, 0.4]))
    assert len(grouped) == 2
    assert grouped[0][0] == (0.0, 0.2)
    assert len(grouped[0][1]) == 2
    assert grouped[1][0] == (0.2, 0.4)
    assert len(grouped[1][1]) == 2


def test_append():
    mol = Molecules(np.zeros((2, 3)), features={"A": [0.1, 0.2], "B": [5, 3]})
    other = Molecules(np.zeros((2, 3)), features={"A": [0.3, 0.4], "B": [2, 1]})
    mol.append(other)
    assert mol.count() == 4
    assert mol.features["B"].to_list() == [5, 3, 2, 1]

    other = Molecules(np.zeros((1, 3)), features={"A": [0.0]})
    mol.append(other)
    assert mol.count() == 5
    assert mol.features["B"].to_list() == [5, 3, 2, 1, None]

    with pytest.raises(ValueError):
        mol.append(Molecules(np.zeros((2, 3)), features={"A": [0.3, 0.4], "C": [2, 1]}))


def test_append_empty():
    mol = Molecules.empty()
    other = Molecules(np.zeros((2, 3)), features={"A": [0.3, 0.4], "B": [2, 1]})
    mol.append(other)
    assert mol.count() == 2
    assert mol.features["B"].to_list() == [2, 1]


@pytest.mark.parametrize(
    "rotvec",
    [
        [0.0, 0.2, 0.0],
        [0.1, 0.1, 0.2],
        [0.1, 0.0, -0.2],
        [0.1, -0.4, 0.3],
        [-1, 0.3, 0.2],
        [-1, -0.3, 0.2],
        [-1, 0.3, -0.2],
        [1, -0.3, -0.2],
        [-0.2, -0.8, -1.3],
        [0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
)
def test_axes_to_rotator(rotvec: list[float]):
    rot = Rotation.from_rotvec(rotvec)
    z = [1, 0, 0]
    y = [0, 1, 0]
    z0 = rot.apply(z)
    y0 = rot.apply(y)
    out = axes_to_rotator(z0, y0)
    assert_allclose(out.as_rotvec(), [rotvec], rtol=1e-8, atol=1e-8)


def test_axes_to_rotator_invert():
    z = [[-1, 0, 0]]
    y = [[0, -1, 0]]
    assert_allclose(
        axes_to_rotator(z, y).as_rotvec(), [[0, 0, np.pi]], rtol=1e-8, atol=1e-8
    )


def test_local_coordinates():
    mol = Molecules(np.zeros((1, 3)), Rotation.random(1))
    coords = mol.local_coordinates((3, 4, 5), squeeze=False)
    assert coords.shape == (1, 3, 3, 4, 5)
    coords = mol.local_coordinates((3, 4, 5), squeeze=True)
    assert coords.shape == (3, 3, 4, 5)


def test_sort():
    pos = np.stack([np.arange(4)] * 3, axis=1)
    mol = Molecules(pos, features={"A": [11, 12, 14, 13]})
    mol_sorted = mol.sort("A")
    assert mol_sorted.features["A"].to_list() == [11, 12, 13, 14]
    assert mol_sorted.pos[:, 0].tolist() == [0, 1, 3, 2]


def test_concat_molecules():
    mole0 = Molecules(np.zeros((4, 3)), features={"A": [1, 2, 3, 4]})
    mole1 = Molecules.empty(feature_labels=["A"])
    mole2 = Molecules(np.ones((2, 3)), features={"A": [5, 6]})
    mole = Molecules.concat([mole0, mole1, mole2], concat_features=True)
    assert mole.count() == 6
    assert_allclose(mole.pos, np.vstack([np.zeros((4, 3)), np.ones((2, 3))]))
    assert mole.features["A"].to_list() == [1, 2, 3, 4, 5, 6]
