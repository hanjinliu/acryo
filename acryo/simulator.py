from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Sequence
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage as ndi
from scipy.fft import fftn, ifftn
from scipy.spatial.transform import Rotation
from dask import array as da
from dask.delayed import delayed

from ._types import nm, pixel, degree
from .molecules import Molecules

from . import _utils

if TYPE_CHECKING:
    from typing_extensions import Self, Literal
    from scipy.spatial.transform import Rotation


class Component(NamedTuple):
    """A component of tomogram."""

    molecules: Molecules
    image: np.ndarray


class TomogramSimulator:
    """
    An object for tomogram simulation

    A TomogramSimulator contains set(s) of molecules and their corresponding density
    and additional information for tomogram generation. For instance, to simulate a
    tomogram with two sets of molecules, you can do

    >>> sim = TomogramSimulator()
    >>> sim.add_molecules(molecules_a, image_a, name="Molecule-A")
    >>> sim.add_molecules(molecules_b, image_b, name="Molecule-B")
    >>> img = sim.simulate(shape=(128, 128, 128))

    Parameters
    ----------
    order : int, default is 3
        Interpolation order for density image.
        - 0 = Nearest neighbor
        - 1 = Linear interpolation
        - 3 = Cubic interpolation
    scale : float, default is 1.0
        Scale of the pixel. This value is used to determine the position of the
        molecules.
    corner_safe : bool, default is False
        Not implemented yet.
    """

    def __init__(
        self,
        order: Literal[0, 1, 3] = 3,
        scale: nm = 1.0,
        corner_safe: bool = False,
    ) -> None:
        # check interpolation order
        if order not in (0, 1, 3):
            raise ValueError(
                f"The third argument 'order' must be 0, 1 or 3, got {order!r}."
            )
        self._order: Literal[0, 1, 3] = order

        # check scale
        self._scale = float(scale)
        if self._scale <= 0:
            raise ValueError("Negative scale is not allowed.")
        self._corner_safe = corner_safe

        self._components: dict[str, Component] = {}

    def __repr__(self) -> str:
        clsname = type(self).__name__
        return f"<{clsname} object with {len(self._components)} components>"

    @property
    def components(self) -> MappingProxyType[str, Component]:
        """Return all the components in the tomogram."""
        return MappingProxyType(self._components)

    @property
    def order(self) -> Literal[0, 1, 3]:
        """Return the interpolation order."""
        return self._order

    @property
    def scale(self) -> float:
        """The physical scale of the tomogram."""
        return self._scale

    @property
    def corner_safe(self) -> bool:
        return self._corner_safe

    def replace(
        self,
        order: Literal[0, 1, 3] | None = None,
        scale: float | None = None,
        corner_safe: bool | None = None,
    ) -> Self:
        """Return a new instance with different parameter(s)."""
        if order is None:
            order = self.order
        if scale is None:
            scale = self.scale
        if corner_safe is None:
            corner_safe = self.corner_safe
        out = self.__class__(
            order=order,
            scale=scale,
            corner_safe=corner_safe,
        )
        out._components = self._components.copy()
        return out

    def copy(self) -> Self:
        """Create a shallow copy of the loader."""
        return self.replace()

    def add_molecules(
        self,
        molecules: Molecules,
        image: np.ndarray,
        *,
        name: str | None = None,
        overwrite: bool = False,
    ) -> Self:
        """
        Add molecules to the tomogram.

        Parameters
        ----------
        molecules : Molecules
            Molecules object that defines the coordinates and orientations of the
            input molecules.
        image : np.ndarray
            Density image of the input molecules.
        name : str, optional
            Name of the molecules., by default None
        overwrite : bool, default is False
            If true, allow overwriting existing component.

        Returns
        -------
        TomogramSimulator
            Same object.
        """
        if name is None:
            name = f"component<{hex(id(molecules))}>"
        if not overwrite and name in self._components:
            raise ValueError(f"Component of name {name!r} already exists.")
        self._components[name] = Component(molecules, image.astype(np.float32))
        return self

    def subset(self, names: str | Sequence[str]) -> Self:
        """
        Construct a simulator composed of a subset of the components.

        Parameters
        ----------
        names : str or sequence of str
            Component names that will be included in the subset.

        Returns
        -------
        TomogramSimulator
            New simulator object.
        """
        if isinstance(names, str):
            _names = {names}
        else:
            _names = set(names)
        new = type(self)(
            order=self.order, scale=self._scale, corner_safe=self.corner_safe
        )
        for name, comp in self._components.items():
            if name in _names:
                new._components[name] = comp
        return new

    def simulate(
        self,
        shape: tuple[pixel, pixel, pixel],
        *,
        tilt_range: tuple[degree, degree] | None = None,
    ) -> NDArray[np.float32]:
        """
        Simulate tomogram.

        Parameters
        ----------
        shape : tuple of int
            Shape of the tomogram.
        tilt_range : tuple of float, optional
            Range of tilt series in degrees, such as (-60., 60.). This argument is
            used to mask missing wedge in the tomogram.

        Returns
        -------
        np.ndarray
            Simulated tomogram image.
        """
        tomogram = np.zeros(shape, dtype=np.float32)
        for mol, image in self._components.values():
            # image slice must be integer so split it into two parts
            pos = mol.pos / self._scale
            intpos = pos.astype(np.int32)
            residue = pos - intpos.astype(np.float32)

            # construct matrices
            template_shape = image.shape
            center = (np.array(template_shape) - 1.0) / 2.0
            starts = intpos - center.astype(np.int32)
            stops = starts + template_shape
            mtxs = _compose_affine_matrices(
                center, mol.rotator.inv(), output_center=center + residue
            )

            # prefilter here to avoid repeated computation
            if self.order > 1:
                image = ndi.spline_filter(image, order=self.order, mode="constant")

            for start, stop, mtx in zip(starts, stops, mtxs):
                # To avoid out-of-boundary, we need to clip the start and stop
                sl_src_list: list[slice] = []
                sl_dst_list: list[slice] = []
                for s, e, size, tsize in zip(start, stop, shape, template_shape):
                    _sl, _pads, _out_of_bound = _utils.make_slice_and_pad(s, e, size)
                    sl_dst_list.append(_sl)
                    if _out_of_bound:
                        s0, s1 = _pads
                        sl_src_list.append(slice(s0, tsize - s1))
                    else:
                        sl_src_list.append(slice(None))
                sl_src = tuple(sl_src_list)
                sl_dst = tuple(sl_dst_list)

                tomogram[sl_dst] += ndi.affine_transform(
                    image,
                    mtx,
                    mode="constant",
                    cval=0.0,
                    order=self.order,
                    prefilter=False,
                )[
                    sl_src
                ]  # type: ignore

        if tilt_range is not None:
            # Remove the missing wedge from the Fourier space
            rot = Rotation.identity()
            mask = _utils.missing_wedge_mask(rot, tilt_range, tomogram.shape)
            ft = fftn(tomogram) * mask  # type: ignore
            tomogram = ifftn(ft).real  # type: ignore

        return tomogram

    def simulate_tilt_series(
        self,
        shape: tuple[pixel, pixel, pixel],
        degrees: Sequence[float],
        central_axis: ArrayLike = (0, 1, 0),
        order: int = 3,
    ) -> NDArray[np.float32]:
        # normalize central axis
        _central_axis = np.asarray(central_axis, dtype=np.float32)
        _central_axis /= np.linalg.norm(_central_axis)

        input = self.simulate(shape)
        matrices = _get_rotation_matrices_for_radon_3d(degrees, _central_axis, shape)
        if order > 1:
            input = ndi.spline_filter(input, order=order, mode="constant")
        tasks = [_radon_transform(input, mtx, order=order) for mtx in matrices]
        out = np.stack(da.compute(tasks)[0], axis=0)  # type: ignore
        return out

    # def simulate_tilt_series_2(
    #     self,
    #     shape: tuple[pixel, pixel],
    #     degrees: Sequence[float],
    #     central_axis: ArrayLike = (0, 1, 0),
    # ) -> NDArray[np.float32]:
    #     # normalize central axis
    #     _central_axis = np.asarray(central_axis, dtype=np.float32)
    #     _central_axis /= np.linalg.norm(_central_axis)
    #     tasks = [
    #         da.from_delayed(
    #             _simulate_single(shape, degree, _central_axis),
    #             shape=shape,
    #             dtype=np.float32
    #         ) for degree in degrees
    #     ]
    #     return da.stack(tasks, axis=0).compute()


def _compose_affine_matrices(
    center: np.ndarray,
    rotator: Rotation,
    output_center: np.ndarray | None = None,
):
    """Compose Affine matrices from an array shape and a Rotation object."""

    dz, dy, dx = center
    if output_center is None:
        raise NotImplementedError

    # center to corner
    translation_0 = np.array(
        [
            [1.0, 0.0, 0.0, dz],
            [0.0, 1.0, 0.0, dy],
            [0.0, 0.0, 1.0, dx],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    # corner to center
    translation_1 = _eyes(len(rotator))
    translation_1[:, :3, 3] = -output_center

    rot_mtx = _eyes(len(rotator))
    rot_mtx[:, :3, :3] = rotator.as_matrix()

    return np.einsum("ij,njk,nkl->nil", translation_0, rot_mtx, translation_1)


def _eyes(n: int) -> np.ndarray:
    return np.stack([np.eye(4, dtype=np.float32)] * n, axis=0)


# @delayed
# def _simulate_single(
#     comp: Component,
#     output_shape: tuple[pixel, pixel],
#     degree: float,
#     central_axis: ArrayLike,
#     order: int = 3,
# ) -> NDArray[np.float32]:
#     mol, image = comp
#     out = np.zeros(output_shape, dtype=np.float32)
#     center = (np.array(image.shape) - 1.0) / 2.0
#     rot = Rotation.from_rotvec(central_axis * np.deg2rad(degree))
#     rot_list = [rot[i] for i in range(len(rot))]
#     matrices = _utils.compose_matrices(center, rot_list, center)

#     for mtx in matrices:
#         ndi.affine_transform(image, mtx, order=order, mode="constant", cval=0.0, prefilter=False)
#     return out


@delayed
def _radon_transform(img: np.ndarray, mtx: np.ndarray, order: int = 3):
    """Radon transform of 3D image."""
    img_rot = ndi.affine_transform(img, mtx, order=order, prefilter=False)
    return np.sum(img_rot, axis=0)


def _get_rotation_matrices_for_radon_3d(
    degrees: Sequence[float],
    central_axis: np.ndarray,
    shape: tuple[int, int, int],
) -> list[NDArray[np.float32]]:
    from scipy.spatial.transform import Rotation

    vec = np.stack([central_axis * np.deg2rad(deg) for deg in degrees], axis=0)
    rot = Rotation.from_rotvec(vec)
    center = (np.array(shape) - 1.0) / 2.0
    return _utils.compose_matrices(center, [rot[i] for i in range(len(rot))], center)
