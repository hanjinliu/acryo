from __future__ import annotations

from types import MappingProxyType
from typing import Callable, NamedTuple, Sequence, TYPE_CHECKING, Tuple
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from ._types import nm, pixel
from .molecules import Molecules

from . import _utils

if TYPE_CHECKING:
    from typing_extensions import Self, Literal
    from scipy.spatial.transform import Rotation
    import pandas as pd

ColorType = Tuple[float, float, float]


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
        colormap=None,
    ) -> NDArray[np.float32]:
        """
        Simulate tomogram.

        Parameters
        ----------
        shape : tuple of int
            Shape of the tomogram.

        Returns
        -------
        np.ndarray
            Simulated tomogram image.
        """
        if len(self._components) == 0:
            import warnings

            warnings.warn("No molecules are added to the simulator.", UserWarning)

        if colormap is not None:
            return self._simulate_with_color(shape, colormap)
        return self._simulate(shape)

    def _simulate(self, shape: tuple[pixel, pixel, pixel]):
        """Simulate a grayscale tomogram."""
        tomogram = np.zeros(shape, dtype=np.float32)
        for mol, image in self._components.values():
            starts, stops, mtxs = _prep_iterators(mol, image, self._scale)

            # prefilter here to avoid repeated computation
            if self.order > 1:
                image = ndi.spline_filter(image, order=self.order, mode="constant")

            for start, stop, mtx in zip(starts, stops, mtxs):
                sl_src, sl_dst = _prep_slices(start, stop, shape, image.shape)

                transformed = ndi.affine_transform(
                    image,
                    mtx,
                    mode="constant",
                    cval=0.0,
                    order=self.order,
                    prefilter=False,
                )
                tomogram[sl_dst] += transformed[sl_src]  # type: ignore

        return tomogram

    def _simulate_with_color(
        self,
        shape: tuple[pixel, pixel, pixel],
        colormap: Callable[[pd.Series], ColorType],
    ):
        """Simulate a colored tomogram."""
        tomogram = np.zeros((3,) + shape, dtype=np.float32)
        for mol, image in self._components.values():
            # image slice must be integer so split it into two parts
            img_min = image.min()
            img_max = image.max()
            starts, stops, mtxs = _prep_iterators(mol, image, self._scale)

            # prefilter here to avoid repeated computation
            if self.order > 1:
                image = ndi.spline_filter(image, order=self.order, mode="constant")

            it = mol.features.iterrows()
            for start, stop, mtx, (_, feat) in zip(starts, stops, mtxs, it):
                _cr, _cg, _cb = colormap(feat)
                sl_src, sl_dst = _prep_slices(start, stop, shape, image.shape)
                sl_dst = (slice(None),) + sl_dst

                transformed = ndi.affine_transform(
                    image,
                    mtx,
                    mode="constant",
                    cval=0.0,
                    order=self.order,
                    prefilter=False,
                )
                _a = (transformed[sl_src] - img_min) / (img_max - img_min)
                color_array = np.stack([_cr * _a, _cg * _a, _cb * _a], axis=0)

                tomogram[sl_dst] += color_array

        return tomogram

    def simulate_2d(self, shape: tuple[pixel, pixel]):
        """Simulate a grayscale tomogram."""
        tomogram = np.zeros(shape, dtype=np.float32)
        for mol, image in self._components.values():
            starts, stops, mtxs = _prep_iterators(mol, image, self._scale)
            zsize = mol.pos[:, 0].max() / self.scale + np.sum(image.shape)
            shape3d = (int(np.ceil(zsize)),) + shape

            # reduce z axis
            starts, stops, mtxs = starts[1:], stops[1:], mtxs[1:]

            # prefilter here to avoid repeated computation
            if self.order > 1:
                image = ndi.spline_filter(image, order=self.order, mode="constant")

            for start, stop, mtx in zip(starts, stops, mtxs):
                sl_src, sl_dst = _prep_slices(start, stop, shape3d, image.shape)
                transformed = ndi.affine_transform(
                    image,
                    mtx,
                    mode="constant",
                    cval=0.0,
                    order=self.order,
                    prefilter=False,
                )
                projected = transformed[sl_src].sum(axis=0)
                tomogram[sl_dst[1:]] += projected  # type: ignore

        return tomogram


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


def _prep_iterators(mol: Molecules, image: np.ndarray, scale: float):

    # image slice must be integer so split it into two parts
    pos = mol.pos / scale
    intpos = pos.astype(np.int32)
    residue = pos - intpos.astype(np.float32)
    template_shape = image.shape

    # construct matrices
    center = (np.array(template_shape) - 1.0) / 2.0
    starts = intpos - center.astype(np.int32)
    stops = starts + template_shape
    mtxs = _compose_affine_matrices(
        center, mol.rotator.inv(), output_center=center + residue
    )

    return starts, stops, mtxs


def _prep_slices(start, stop, tomogram_shape, template_shape):
    sl_src_list: list[slice] = []
    sl_dst_list: list[slice] = []
    for s, e, size, tsize in zip(start, stop, tomogram_shape, template_shape):
        _sl, _pads, _out_of_bound = _utils.make_slice_and_pad(s, e, size)
        sl_dst_list.append(_sl)
        if _out_of_bound:
            s0, s1 = _pads
            sl_src_list.append(slice(s0, tsize - s1))
        else:
            sl_src_list.append(slice(None))
    sl_src = tuple(sl_src_list)
    sl_dst = tuple(sl_dst_list)
    return sl_src, sl_dst
