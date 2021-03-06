from __future__ import annotations
from typing import Iterable, TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation
from ._types import nm

if TYPE_CHECKING:
    import pandas as pd

_CSV_COLUMNS = ["z", "y", "x", "zvec", "yvec", "xvec"]


class Molecules:
    """
    Object that represents position- and orientation-defined molecules.

    Positions are represented by a (N, 3) ``np.ndarray`` and orientations
    are represented by a ``scipy.spatial.transform.Rotation`` object.
    Features of each molecule can also be recorded by the ``features``
    property.

    Parameters
    ----------
    pos : ArrayLike
        Moleculs positions.
    rot : scipy.spatial.transform.Rotation object
        Molecule orientations.
    features : dataframe, optional
        Molecule features.
    """

    def __init__(
        self,
        pos: ArrayLike,
        rot: Rotation | None = None,
        features: pd.DataFrame | ArrayLike | dict[str, ArrayLike] | None = None,
    ):
        pos = np.atleast_2d(pos)

        if pos.shape[1] != 3:
            raise ValueError("Shape of pos must be (N, 3).")

        if rot is None:
            quat = np.stack([np.array([0, 0, 0, 1])] * pos.shape[0], axis=0)
            rot = Rotation.from_quat(quat)
        elif pos.shape[0] != len(rot):
            raise ValueError(
                f"Length mismatch. There are {pos.shape[0]} molecules but {len(rot)} "
                "rotation were given."
            )

        self._pos = pos
        self._rotator = rot
        self._features: pd.DataFrame | None = None
        self.features = features

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n={len(self)})"

    @classmethod
    def from_axes(
        cls,
        pos: np.ndarray,
        z: np.ndarray | None = None,
        y: np.ndarray | None = None,
        x: np.ndarray | None = None,
    ) -> Molecules:
        """Construct molecule cloud with orientation from two of their local axes."""
        pos = np.atleast_2d(pos)

        if sum((_ax is not None) for _ax in [z, y, x]) != 2:
            raise TypeError("You must specify two out of z, y, and x.")

        # NOTE: np.cross assumes vectors are in xyz order. However, all the arrays here
        # are defined in zyx order. To build right-handed coordinates, we must invert
        # signs when using np.cross.
        if z is None:
            if x is None or y is None:
                raise TypeError("Two of x, y, z is needed.")
            x = np.atleast_2d(x)
            y = np.atleast_2d(y)
            z = -np.cross(x, y, axis=1)
        elif y is None:
            if x is None or z is None:
                raise TypeError("Two of x, y, z is needed.")
            z = np.atleast_2d(z)
            x = np.atleast_2d(x)
            y = -np.cross(z, x, axis=1)

        rotator = axes_to_rotator(z, y)
        return cls(pos, rotator)

    @classmethod
    def from_euler(
        cls,
        pos: np.ndarray,
        angles: ArrayLike,
        seq: str = "ZXZ",
        degrees: bool = False,
        order: str = "xyz",
        features: pd.DataFrame | None = None,
    ):
        """Create molecules from Euler angles."""
        if order == "xyz":
            rotator = from_euler_xyz_coords(angles, seq, degrees)
        elif order == "zyx":
            rotator = Rotation.from_euler(seq, angles, degrees)
        else:
            raise ValueError("'order' must be 'xyz' or 'zyx'.")
        return cls(pos, rotator, features)

    @classmethod
    def from_csv(
        cls,
        path: str,
        pos_cols: list[str] = ["z", "y", "x"],
        rot_cols: list[str] = ["zvec", "yvec", "xvec"],
        **pd_kwargs,
    ) -> Molecules:
        """Load csv as a Molecules object."""
        import pandas as pd

        pos_cols = pos_cols.copy()
        rot_cols = rot_cols.copy()
        df: pd.DataFrame = pd.read_csv(path, **pd_kwargs)  # type: ignore
        pos = df[pos_cols]
        rotvec = df[rot_cols]
        cols = pos + rotvec
        others = df.iloc[:, np.array([c not in cols for c in df.columns])]
        return cls(
            pos,
            Rotation.from_rotvec(rotvec),
            features=others,
        )

    @property
    def features(self) -> pd.DataFrame:
        """Molecules features."""
        if self._features is None:
            import pandas as pd

            return pd.DataFrame(None)
        return self._features

    @features.setter
    def features(self, value):
        if value is None:
            self._features = None
        else:
            import pandas as pd

            df = pd.DataFrame(value)
            if len(df) != self.pos.shape[0]:
                raise ValueError(
                    f"Length mismatch. There are {self.pos.shape[0]} molecules but "
                    f"{len(df)} features were given."
                )
            self._features = df
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert coordinates, rotation and features into a single data frame."""
        import pandas as pd

        rotvec = self.rotvec()
        data = np.concatenate([self.pos, rotvec], axis=1)
        df = pd.DataFrame(data, columns=_CSV_COLUMNS)
        if self._features is not None:
            df = pd.concat([df, self._features], axis=1)
        return df

    def to_csv(self, save_path: str) -> None:
        """
        Save molecules as a csv file.

        Parameters
        ----------
        save_path : str
            Save path.
        """
        return self.to_dataframe().to_csv(save_path, index=False)

    def __len__(self) -> int:
        """Return the number of molecules."""
        return self._pos.shape[0]

    def __getitem__(self, key: int | slice | list[int] | np.ndarray) -> Molecules:
        return self.subset(key)

    @property
    def pos(self) -> np.ndarray:
        """Positions of molecules."""
        return self._pos

    @property
    def x(self) -> np.ndarray:
        """Vectors of x-axis."""
        return self._rotator.apply([0.0, 0.0, 1.0])

    @property
    def y(self) -> np.ndarray:
        """Vectors of y-axis."""
        return self._rotator.apply([0.0, 1.0, 0.0])

    @property
    def z(self) -> np.ndarray:
        """Vectors of z-axis."""
        return self._rotator.apply([1.0, 0.0, 0.0])

    @property
    def rotator(self) -> Rotation:
        """Return ``scipy.spatial.transform.Rotation`` object"""
        return self._rotator

    @classmethod
    def concat(
        cls, moles: Iterable[Molecules], concat_features: bool = True
    ) -> Molecules:
        """Concatenate Molecules objects."""
        pos: list[np.ndarray] = []
        quat: list[np.ndarray] = []
        features: list[pd.DataFrame | None] = []
        for mol in moles:
            pos.append(mol.pos)
            quat.append(mol.quaternion())
            features.append(mol._features)

        all_pos = np.concatenate(pos, axis=0)
        all_quat = np.concatenate(quat, axis=0)
        if concat_features:
            import pandas as pd

            all_features = pd.concat(features, axis=0)
        else:
            all_features = None

        return cls(all_pos, Rotation(all_quat), features=all_features)

    def subset(self, spec: int | slice | list[int] | np.ndarray) -> Molecules:
        """
        Create a subset of molecules by slicing.

        Any slicing supported in ``numpy.ndarray``, except for integer, can be
        used here. Molecule positions and angles are sliced at the same time.

        Parameters
        ----------
        spec : int, slice, list of int, or ndarray
            Specifier that defines which molecule will be used. Any objects
            that numpy slicing are defined are supported. For instance,
            ``[2, 3, 5]`` means the 2nd, 3rd and 5th molecules will be used
            (zero-indexed), and ``slice(10, 20)`` means the 10th to 19th
            molecules will be used.

        Returns
        -------
        Molecules
            Molecule subset.
        """
        if isinstance(spec, int):
            spec = slice(spec, spec + 1)
        pos = self.pos[spec]
        quat = self._rotator.as_quat()[spec]
        if self._features is None:
            return self.__class__(pos, Rotation(quat))
        return self.__class__(pos, Rotation(quat), self._features.iloc[spec, :])

    def affine_matrix(
        self, src: np.ndarray, dst: np.ndarray | None = None, inverse: bool = False
    ) -> np.ndarray:
        """
        Construct affine matrices using positions and angles of molecules.

        Parameters
        ----------
        src : np.ndarray
            Source coordinates.
        dst : np.ndarray, optional
            Destination coordinates. By default the coordinates of molecules will be used.
        inverse : bool, default is False
            Return inverse mapping if true.

        Returns
        -------
        (N, 4, 4) array
            Array of concatenated affine matrices.
        """
        if dst is None:
            dst = self.pos

        nmole = len(self)

        if inverse:
            mat = self._rotator.inv().as_matrix()
        else:
            mat = self.matrix()
        rot_mat = np.zeros((nmole, 4, 4), dtype=np.float32)
        rot_mat[:, 3, 3] = 1.0
        rot_mat[:, :3, :3] = mat

        translation_0 = np.stack([np.eye(4, dtype=np.float32)] * nmole, axis=0)
        translation_1 = np.stack([np.eye(4, dtype=np.float32)] * nmole, axis=0)
        translation_0[:, :3, 3] = dst
        translation_1[:, :3, 3] = -src

        return np.einsum("nij,njk,nkl->nil", translation_0, rot_mat, translation_1)

    def cartesian_at(
        self,
        index: int | slice | Iterable[int],
        shape: tuple[int, int, int],
        scale: nm,
    ) -> np.ndarray:
        if isinstance(index, int):
            center = np.array(shape) / 2 - 0.5
            vec_x = self._rotator[index].apply([0.0, 0.0, 1.0])
            vec_y = self._rotator[index].apply([0.0, 1.0, 0.0])
            vec_z = -np.cross(vec_x, vec_y)
            ind_z, ind_y, ind_x = [np.arange(s) - c for s, c in zip(shape, center)]
            x_ax: np.ndarray = vec_x[:, np.newaxis] * ind_x
            y_ax: np.ndarray = vec_y[:, np.newaxis] * ind_y
            z_ax: np.ndarray = vec_z[:, np.newaxis] * ind_z

            # There will be many points so data type should be converted into 32-bit
            x_ax = x_ax.astype(np.float32)
            y_ax = y_ax.astype(np.float32)
            z_ax = z_ax.astype(np.float32)

            coords = (
                z_ax[:, :, np.newaxis, np.newaxis]
                + y_ax[:, np.newaxis, :, np.newaxis]
                + x_ax[:, np.newaxis, np.newaxis, :]
            )
            shifts = self.pos[index] / scale
            coords += shifts[:, np.newaxis, np.newaxis, np.newaxis]  # unit: pixel
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            coords = np.stack(
                [self.cartesian_at(i, shape, scale) for i in range(start, stop, step)],
                axis=0,
            )
        else:
            coords = np.stack(
                [self.cartesian_at(i, shape, scale) for i in index],
                axis=0,
            )

        return coords

    def matrix(self) -> np.ndarray:
        """
        Calculate rotation matrices that align molecules in such orientations
        that ``vec`` belong to the object.

        Returns
        -------
        (N, 3, 3) ndarray
            Rotation matrices. Rotations represented by these matrices transform
            molecules to the same orientations, i.e., align all the molecules.
        """
        return self._rotator.as_matrix()

    def euler_angle(self, seq: str = "ZXZ", degrees: bool = False) -> np.ndarray:
        """
        Calculate Euler angles that transforms a source vector to vectors that
        belong to the object.

        Parameters
        ----------
        seq : str, default is "ZXZ"
            Copy of ``scipy.spatial.transform.Rotation.as_euler``. 3 characters
            belonging to the set {"X", "Y", "Z"} for intrinsic rotations, or
            {"x", "y", "z"} for extrinsic rotations. Adjacent axes cannot be the
            same. Extrinsic and intrinsic rotations cannot be mixed in one function
            call.
        degrees: bool, default is False
            Copy of ``scipy.spatial.transform.Rotation.as_euler``. Returned
            angles are in degrees if this flag is True, else they are in radians.

        Returns
        -------
        (N, 3) ndarray
            Euler angles.
        """
        seq = _translate_euler(seq)
        return self._rotator.as_euler(seq, degrees=degrees)[..., ::-1]

    def quaternion(self) -> np.ndarray:
        """
        Calculate quaternions that transforms a source vector to vectors that
        belong to the object.

        Returns
        -------
        (N, 4) ndarray
            Quaternions.
        """
        return self._rotator.as_quat()

    def rotvec(self) -> np.ndarray:
        """
        Calculate rotation vectors that transforms a source vector to vectors
        that belong to the object.

        Returns
        -------
        (N, 3) ndarray
            Rotation vectors.
        """
        return self._rotator.as_rotvec()

    def translate(self, shifts: ArrayLike, copy: bool = True) -> Molecules:
        """
        Translate molecule positions by ``shifts``.

        Shifts are applied in world coordinates, not internal coordinates of
        every molecules. If molecules should be translated in their own
        coordinates, such as translating toward y-direction of each molecules
        by 1.0 nm, use ``translate_internal`` instead. Translation operation
        does not convert molecule orientations.

        Parameters
        ----------
        shifts : (3,) or (N, 3) array
            Spatial shift of molecules.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing
            instance.

        Returns
        -------
        Molecules
            Instance with updated positional coordinates.
        """
        coords = self._pos + shifts
        if copy:
            features = self._features
            if features is not None:
                features = features.copy()
            out = self.__class__(coords, self._rotator, features=features)
        else:
            self._pos = coords
            out = self
        return out

    def translate_internal(self, shifts: ArrayLike, *, copy: bool = True) -> Molecules:
        """
        Translate molecule positions internally by ``shifts``.

        Shifts are applied in world coordinates, not internal coordinates of
        every molecules. If molecules should be translated in their own
        coordinates, such as translating toward y-direction of each molecules
        by 1.0 nm, use ``translate_internal`` instead. Translation operation
        does not convert molecule orientations.

        Parameters
        ----------
        shifts : (3,) or (N, 3) array
            Spatial shift of molecules.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing
            instance.

        Returns
        -------
        Molecules
            Instance with updated positional coordinates.
        """
        world_shifts = self._rotator.apply(shifts)
        return self.translate(world_shifts, copy=copy)

    def translate_random(
        self,
        max_distance: nm,
        *,
        seed: int | None = None,
        copy: bool = True,
    ) -> Molecules:
        """
        Apply random translation to each molecule.

        Translation range is restricted by a maximum distance and translation
        values are uniformly distributed in this region. Different translations
        will be applied to different molecules.

        Parameters
        ----------
        max_distance : nm
            Maximum distance of translation.
        seed : int, optional
            Random seed, by default None
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing
            instance.

        Returns
        -------
        Molecules
            Translated molecules.
        """
        nmole = len(self)
        np.random.seed(seed)
        r = np.random.random(nmole) * max_distance
        theta = np.random.random(nmole) * 2 * np.pi
        phi = np.random.random(nmole) * np.pi
        np.random.seed(None)
        cos_phi = np.cos(phi)
        shifts = np.stack(
            [
                r * np.sin(phi),
                r * cos_phi * np.sin(theta),
                r * cos_phi * np.cos(theta),
            ],
            axis=1,
        )
        return self.translate(shifts, copy=copy)

    def rotate_by_rotvec_internal(
        self, vector: ArrayLike, copy: bool = True
    ) -> Molecules:
        """
        Rotate molecules using internal rotation vector.

        Vector components are calculated in the molecule-coordinate.

        Parameters
        ----------
        vector : ArrayLike
            Rotation vector(s).
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing instance.

        Returns
        -------
        Molecules
            Instance with updated angles.
        """
        vector = np.atleast_2d(vector)
        vec_x = self.x
        vec_y = self.y
        vec_z = -np.cross(vec_x, vec_y, axis=1)
        world_rotvec = (
            vec_z * vector[:, 0][:, np.newaxis]
            + vec_y * vector[:, 1][:, np.newaxis]
            + vec_x * vector[:, 2][:, np.newaxis]
        )
        return self.rotate_by_rotvec(world_rotvec, copy=copy)

    def rotate_by_matrix(self, matrix: ArrayLike, copy: bool = True) -> Molecules:
        """
        Rotate molecules using rotation matrices, **with their position unchanged**.

        Parameters
        ----------
        matrix : ArrayLike
            Rotation matrices, whose length must be same as the number of
            molecules.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing
            instance.

        Returns
        -------
        Molecules
            Instance with updated orientation.
        """
        rotator = Rotation.from_matrix(matrix)
        return self.rotate_by(rotator, copy)

    def rotate_by_quaternion(self, quat: ArrayLike, copy: bool = True) -> Molecules:
        """
        Rotate molecules using quaternions, **with their position unchanged**.

        Parameters
        ----------
        quat : ArrayLike
            Rotation quaternion.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing
            instance.

        Returns
        -------
        Molecules
            Instance with updated orientation.
        """
        rotator = Rotation.from_quat(quat)
        return self.rotate_by(rotator, copy)

    def rotate_by_euler_angle(
        self,
        angles: ArrayLike,
        seq: str = "ZXZ",
        degrees: bool = False,
        order: str = "xyz",
        copy: bool = True,
    ) -> Molecules:
        """
        Rotate molecules using Euler angles, **with their position unchanged**.

        Parameters
        ----------
        angles: array-like
            Euler angles of rotation.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing
            instance.

        Returns
        -------
        Molecules
            Instance with updated orientation.
        """
        if order == "xyz":
            rotator = from_euler_xyz_coords(angles, seq, degrees)
        elif order == "zyx":
            rotator = Rotation.from_euler(seq, angles, degrees)
        else:
            raise ValueError("'order' must be 'xyz' or 'zyx'.")
        return self.rotate_by(rotator, copy)

    def rotate_by_rotvec(self, vector: ArrayLike, copy: bool = True) -> Molecules:
        """
        Rotate molecules using rotation vectors, **with their position unchanged**.

        Parameters
        ----------
        vector: array-like
            Rotation vectors.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing
            instance.

        Returns
        -------
        Molecules
            Instance with updated orientation.
        """
        rotator = Rotation.from_rotvec(vector)
        return self.rotate_by(rotator, copy)

    def rotate_by(self, rotator: Rotation, copy: bool = True) -> Molecules:
        """
        Rotate molecule with a ``Rotation`` object.

        Note that ``Rotation`` object satisfies following equation.

        .. code-block::python

            rot1.apply(rot2.apply(v)) == (rot1*rot2).apply(v)

        Parameters
        ----------
        rotator : Rotation
            Molecules will be rotated by this object.
        copy : bool, default is True
            If true, create a new instance, otherwise overwrite the existing instance.

        Returns
        -------
        Molecules
            Instance with updated orientation.
        """
        rot = rotator * self._rotator
        if copy:
            features = self._features
            if features is not None:
                features = features.copy()
            out = self.__class__(self._pos, rot, features=features)
        else:
            self._rotator = rot
            out = self
        return out

    def linear_transform(
        self,
        shift: ArrayLike,
        rotator: Rotation,
        inv: bool = False,
    ) -> Molecules:
        """Shift and rotate molecules around their own coordinate."""
        rotvec = rotator.as_rotvec()
        if inv:
            shift_corrected = rotator.apply(shift, inverse=True)
            return self.rotate_by_rotvec_internal(-rotvec).translate_internal(
                shift_corrected
            )
        else:
            shift_corrected = rotator.apply(shift)
            return self.translate_internal(shift_corrected).rotate_by_rotvec_internal(
                rotvec
            )


def _translate_euler(seq: str) -> str:
    table = str.maketrans({"x": "z", "z": "x", "X": "Z", "Z": "X"})
    return seq[::-1].translate(table)


def from_euler_xyz_coords(
    angles: ArrayLike, seq: str = "ZXZ", degrees: bool = False
) -> Rotation:
    """Create a rotator using zyx-coordinate system, from Euler angles."""
    seq = _translate_euler(seq)
    angles = np.asarray(angles)
    return Rotation.from_euler(seq, angles[..., ::-1], degrees)


def _normalize(a: np.ndarray) -> np.ndarray:
    """Normalize vectors to length 1. Input must be (N, 3)."""
    return a / np.sqrt(np.sum(a ** 2, axis=1))[:, np.newaxis]


def _extract_orthogonal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Extract component of b orthogonal to a."""
    a_norm = _normalize(a)
    return b - np.sum(a_norm * b, axis=1)[:, np.newaxis] * a_norm


def axes_to_rotator(z, y) -> Rotation:
    ref = _normalize(np.atleast_2d(y))

    n = ref.shape[0]
    yx = np.arctan2(ref[:, 2], ref[:, 1])
    zy = np.arctan(-ref[:, 0] / np.abs(ref[:, 1]))

    rot_vec_yx = np.zeros((n, 3))
    rot_vec_yx[:, 0] = yx
    rot_yx = Rotation.from_rotvec(rot_vec_yx)

    rot_vec_zy = np.zeros((n, 3))
    rot_vec_zy[:, 2] = zy
    rot_zy = Rotation.from_rotvec(rot_vec_zy)

    rot1 = rot_yx * rot_zy

    if z is None:
        return rot1

    vec = _normalize(np.atleast_2d(_extract_orthogonal(ref, z)))

    vec_trans = rot1.apply(vec, inverse=True)  # in zx-plane

    thetas = np.arctan2(vec_trans[..., 0], vec_trans[..., 2]) - np.pi / 2

    rot_vec_zx = np.zeros((n, 3))
    rot_vec_zx[:, 1] = thetas
    rot2 = Rotation.from_rotvec(rot_vec_zx)

    return rot1 * rot2
