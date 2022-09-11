from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Sequence
import numpy as np
from scipy import ndimage as ndi
from ._types import nm, pixel
from .molecules import Molecules

from . import _utils

if TYPE_CHECKING:
    from typing_extensions import Self
    from scipy.spatial.transform import Rotation


class Component(NamedTuple):
    molecules: Molecules
    image: np.ndarray


class TomogramSimulator:
    def __init__(
        self,
        order: int = 3,
        scale: nm = 1.0,
        corner_safe: bool = False,
    ) -> None:
        # check interpolation order
        if order not in (0, 1, 3):
            raise ValueError(
                f"The third argument 'order' must be 0, 1 or 3, got {order!r}."
            )
        self._order = order

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
    def order(self) -> int:
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
        order: int | None = None,
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

    def simulate(self, shape: tuple[pixel, pixel, pixel]) -> np.ndarray:
        tomogram = np.zeros(shape, dtype=np.float32)
        for mol, image in self._components.values():
            pos = mol.pos / self._scale
            intpos = pos.astype(np.int32)
            residue = pos - intpos.astype(np.float32)
            template_shape = image.shape
            center = (np.array(template_shape) - 1.0) / 2.0
            starts = intpos - center.astype(np.int32)
            stops = starts + template_shape
            mtxs = _compose_affine_matrices(
                center, mol.rotator, output_center=center + residue
            )

            # prefilter here to avoid repeated computation
            if self.order > 1:
                image = ndi.spline_filter(image, order=self.order, mode="constant")

            for start, stop, mtx in zip(starts, stops, mtxs):
                # To avoid out-of-boundary, we need to clip the start and stop
                sl_src: list[slice] = []
                sl_dst: list[slice] = []
                for s, e, size, tsize in zip(start, stop, shape, template_shape):
                    _sl, _pads, _out_of_bound = _utils.make_slice_and_pad(s, e, size)
                    sl_dst.append(_sl)
                    if _out_of_bound:
                        s0, s1 = _pads
                        sl_src.append(slice(s0, tsize - s1))
                    else:
                        sl_src.append(slice(None))

                tomogram[tuple(sl_dst)] += ndi.affine_transform(
                    image,
                    mtx,
                    mode="constant",
                    cval=0.0,
                    order=self.order,
                    prefilter=False,
                )[
                    tuple(sl_src)
                ]  # type: ignore

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
