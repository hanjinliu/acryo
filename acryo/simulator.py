# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import warnings
from types import MappingProxyType
from typing import (
    Callable,
    Iterable,
    NamedTuple,
    Sequence,
    TYPE_CHECKING,
    Union,
    TypeVar,
)
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from acryo import _utils
from acryo.molecules import Molecules, axes_to_rotator
from acryo.pipe._classes import ImageProvider
from acryo._types import nm, pixel
from acryo._dask import DaskTaskPool
from acryo._typed_scipy import spline_filter, affine_transform

if TYPE_CHECKING:
    from typing_extensions import Self, Literal
    from scipy.spatial.transform import Rotation
    import polars as pl

ColorType = Union[tuple[float, float, float], tuple[float, float, float, float]]
ColorFunc = Callable[["pl.DataFrame"], ColorType]


class Component(NamedTuple):
    """A component of tomogram."""

    molecules: Molecules
    image: NDArray[np.float32] | ImageProvider


class TomogramSimulator:
    """An object for tomogram simulation

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
        *,
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
        return MappingProxyType(self._components)  # type: ignore

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
        image: NDArray[np.float32] | ImageProvider,
        *,
        name: str | None = None,
        overwrite: bool = False,
    ) -> Self:
        """Add molecules to the tomogram.

        Parameters
        ----------
        molecules : Molecules
            Molecules object that defines the coordinates and orientations of the
            input molecules.
        image : np.ndarray or ImageProvider
            Density image of the input molecules, or an image provider.
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
        if isinstance(image, np.ndarray):
            if image.dtype != np.float32:
                image = image.astype(np.float32)
        elif not isinstance(image, ImageProvider):
            raise TypeError(
                f"The second argument must be an array or an ImageProvider, got {type(image)}."
            )
        self._components[name] = Component(molecules, image)
        return self

    def collect_molecules(self) -> Molecules:
        return Molecules.concat(comp.molecules for comp in self._components.values())

    def subset(self, names: str | Sequence[str]) -> Self:
        """Construct a simulator composed of a subset of the components.

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
        colormap: ColorFunc | NDArray[np.float32] | None = None,
    ) -> NDArray[np.float32]:
        """Simulate a tomogram.

        Parameters
        ----------
        shape : tuple of int
            Shape of the tomogram.
        colormap: callable, optional
            Colormap used to generate the colored tomogram. The input is a polars
            row vector of features of each molecule.

        Returns
        -------
        np.ndarray
            Simulated tomogram image.
        """
        if len(self._components) == 0:
            warnings.warn("No molecules are added to the simulator.", UserWarning)

        if colormap is not None:
            return self._simulate_with_color(shape, colormap)
        if len(shape) != 3:
            raise ValueError("The shape must be a 3-tuple.")
        return self._simulate(shape)

    def _simulate(self, shape: tuple[pixel, pixel, pixel]):
        """Simulate a grayscale tomogram."""
        tomogram = np.zeros(shape, dtype=np.float32)
        pool = DaskTaskPool.from_func(_simulate_one)
        for mol, image in self._components.values():
            img = self._get_image(image)
            starts, stops, mtxs = _prep_iterators(mol, img.shape, self._scale)
            for start, stop, mtx in zip(starts, stops, mtxs):
                pool.add_task(img, start, stop, mtx, shape, self.order)

        results = pool.compute()
        for sl, img_fragment in results:
            if img_fragment is not None:
                tomogram[sl] += img_fragment

        return tomogram

    def _simulate_with_color(
        self,
        shape: tuple[pixel, pixel, pixel],
        colormap: ColorFunc | NDArray[np.float32],
    ):
        """Simulate a colored tomogram."""
        if isinstance(colormap, np.ndarray):
            if len(self._components) != 1:
                raise ValueError(
                    "If an array is given as the colormap, cannot simulate colored tomogram"
                )
            if colormap.shape[1] not in (3, 4):
                raise ValueError(f"Wrong shape as a colormap {colormap.shape}.")
            cmap = lambda df, i: colormap[i]
        else:
            cmap = lambda df, i: colormap(df[i])

        tomogram = np.zeros((3,) + shape, dtype=np.float32)
        pool = DaskTaskPool.from_func(_simulate_color_one)
        for mol, image in self._components.values():
            img = self._get_image(image)
            starts, stops, mtxs = _prep_iterators(mol, img.shape, self._scale)
            feat = mol.features
            for i, (start, stop, mtx) in enumerate(zip(starts, stops, mtxs)):
                f0 = cmap(feat, i)
                # f0 = colormap(feat[i])
                pool.add_task(img, start, stop, mtx, shape, self.order, f0)

        results = pool.compute()
        for sl, img_fragment in results:
            if img_fragment is not None:
                tomogram[sl] += img_fragment

        return tomogram

    def simulate_2d(self, shape: tuple[pixel, pixel]) -> NDArray[np.float32]:
        """Simulate a grayscale tomogram."""
        projection = np.zeros(shape, dtype=np.float32)
        pool = DaskTaskPool.from_func(_simulate_2d_one)
        for mol, image in self._components.values():
            img = self._get_image(image)
            starts, stops, mtxs = _prep_iterators(mol, img.shape, self._scale)
            zsize = mol.pos[:, 0].max() / self.scale + np.sum(img.shape)
            shape3d = (int(np.ceil(zsize)),) + shape

            # reduce z axis
            starts, stops, mtxs = starts[1:], stops[1:], mtxs[1:]

            for start, stop, mtx in zip(starts, stops, mtxs):
                pool.add_task(img, start, stop, mtx, shape3d, self.order)

        results = pool.compute()
        for sl, img_fragment in results:
            if img_fragment is not None:
                projection[sl] += img_fragment

        return projection

    def simulate_projection(
        self,
        shape: tuple[int, int],
        center: tuple[float, float, float],
        xaxis: tuple[float, float, float],
        yaxis: tuple[float, float, float],
    ) -> NDArray[np.float32]:
        """Simulate a projection of the tomogram in any direction.

        Projection plane and the projection range are defined by the given parameters.
        ``shape`` and ``center`` define the range of the projection plane. Technically,
        ``center`` will be the scale-considered center of the assumed 3D tomogram image.
        ``xaxis`` and ``yaxis`` define the coordinate system of the projection plane.

        Parameters
        ----------
        shape : (int, int)
            Output shape of the projection.
        center : (float, float, float)
            Center of the projection plane in world coordinate system.
        xaxis : (float, float, float)
            X-axis of the projection plane in world coordinate system.
        yaxis : (float, float, float)
            Y-axis of the projection plane in world coordinate system.

        Returns
        -------
        2D array
            Simulated projection image.
        """
        projection = np.zeros(shape, dtype=np.float32)
        ex = np.asarray(xaxis, dtype=np.float32) / np.linalg.norm(xaxis)
        ey = np.asarray(yaxis, dtype=np.float32) / np.linalg.norm(yaxis)
        if np.abs(np.dot(ex, ey)) > 1e-6:
            raise ValueError(f"xaxis {xaxis!r} and yaxis {yaxis!r} are not orthogonal.")
        rc = np.asarray(center, dtype=np.float32)
        pool = DaskTaskPool.from_func(_simulate_projection_one)
        for mol, image in self._components.values():
            img = self._get_image(image)
            pos_scaled = (mol.pos - rc) / self.scale
            xcoords = pos_scaled.dot(ex) + (shape[1] - 1) / 2
            ycoords = pos_scaled.dot(ey) + (shape[0] - 1) / 2
            coords = np.stack((ycoords, xcoords), axis=1)
            glob_rotator = axes_to_rotator(cross(ex, ey), ey)
            for i, yx in enumerate(coords):
                pool.add_task(yx, shape, img, mol.rotator[i], glob_rotator)

        results = pool.compute()
        for sl, img_fragment in results:
            if img_fragment is not None:
                projection[sl] += img_fragment

        return projection

    def simulate_tilt_series(
        self,
        degrees: Iterable[float],
        shape: tuple[int, int, int],
    ) -> NDArray[np.float32]:
        """Simulate a tilt series of the tomogram.

        This method is a simplified version of :meth:`simulate_projection`. Projection
        angles are restricted to the rotation around the y-axis of the world coordinate
        and the tomogram will be projected parallel to the z-axis.

        Parameters
        ----------
        degrees : iterable of float
            Rotation angles.
        shape : (int, int, int)
            Tomogram shape.

        Returns
        -------
        (N, Y, X) array
            Tilting series image.
        """
        rads = np.deg2rad(list(degrees))
        ntilt = len(rads)
        ey = np.array([0, 1, 0], dtype=np.float32)
        rc = (np.array(shape, dtype=np.float32) / 2 - 0.5) * self.scale
        tilt_series = np.zeros((ntilt, shape[1], shape[2]), dtype=np.float32)
        pool = DaskTaskPool.from_func(_simulate_projection_one_labeled)
        for i, rad in enumerate(rads):
            ex = np.array([np.sin(rad), 0, np.cos(rad)], dtype=np.float32)
            for mol, image in self._components.values():
                img = self._get_image(image)
                pos_scaled = (mol.pos - rc) / self.scale
                xcoords = pos_scaled.dot(ex) + (shape[2] - 1) / 2
                ycoords = pos_scaled.dot(ey) + (shape[1] - 1) / 2
                coords = np.stack((ycoords, xcoords), axis=1)
                glob_rotator = axes_to_rotator(cross(ex, ey), ey)
                for ci, yx in enumerate(coords):
                    pool.add_task(yx, shape[1:], img, mol.rotator[ci], glob_rotator, i)

        results = pool.compute()
        for sl, img_fragment in results:
            if img_fragment is not None:
                tilt_series[sl] += img_fragment

        return tilt_series

    def _get_image(
        self, image: NDArray[np.float32] | ImageProvider
    ) -> NDArray[np.float32]:
        if isinstance(image, ImageProvider):
            img = image.provide(self.scale)
        else:
            img = image

        if self.order > 1:
            img = spline_filter(img, order=self.order, mode="constant")
        return img


def _compose_affine_matrices(
    center: np.ndarray,
    rotator: Rotation,
    output_center: np.ndarray | None = None,
) -> NDArray[np.float32]:
    """Compose Affine matrices from an array shape and a Rotation object."""
    if output_center is None:
        raise NotImplementedError

    dz, dy, dx = center

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

    return np.einsum("ij,njk,nkl->nil", translation_0, rot_mtx, translation_1)  # type: ignore


def _eyes(n: int) -> np.ndarray:
    return np.stack([np.eye(4, dtype=np.float32)] * n, axis=0)


def _prep_iterators(mol: Molecules, shape: tuple[int, int, int], scale: float):
    # image slice must be integer so split it into two parts
    pos = mol.pos / scale
    intpos = pos.astype(np.int32)
    residue = pos - intpos.astype(np.float32)

    # construct matrices
    center = (np.array(shape) - 1.0) / 2.0
    starts = intpos - center.astype(np.int32)
    stops = starts + shape
    mtxs = _compose_affine_matrices(
        center, mol.rotator.inv(), output_center=center + residue
    )

    return starts, stops, mtxs


_T = TypeVar("_T", bound=Iterable[int])


def _prep_slices(
    start: _T,
    stop: _T,
    tomogram_shape: _T,
    template_shape: _T,
) -> tuple[tuple[slice, ...], tuple[slice, ...] | None]:
    sl_src_list: list[slice] = []
    sl_dst_list: list[slice] = []
    for s, e, size, tsize in zip(start, stop, tomogram_shape, template_shape):
        try:
            _sl, _pads, _out_of_bound = _utils.make_slice_and_pad(s, e, size)
        except ValueError:
            # out-of-bound is not a problem while simulation
            return (slice(None),), None
        else:
            sl_dst_list.append(_sl)
            if _out_of_bound:
                s0, s1 = _pads
                sl_src_list.append(slice(s0, tsize - s1))
            else:
                sl_src_list.append(slice(None))
    sl_src = tuple(sl_src_list)
    sl_dst = tuple(sl_dst_list)
    return sl_src, sl_dst


def _simulate_one(
    img: NDArray[np.float32],
    start: tuple[int, int, int],
    stop: tuple[int, int, int],
    mtx: NDArray[np.float32],
    shape: tuple[int, int, int],
    order: int,
) -> tuple[tuple[slice, ...], NDArray[np.float32] | None]:
    sl_src, sl_dst = _prep_slices(start, stop, shape, img.shape)
    if sl_dst is None:
        return (slice(None),), None
    transformed = affine_transform(
        img,
        mtx,
        mode="constant",
        cval=0.0,
        order=order,
        prefilter=False,
    )
    return sl_dst, transformed[sl_src]


def _simulate_color_one(
    img: NDArray[np.float32],
    start: tuple[int, int, int],
    stop: tuple[int, int, int],
    mtx: NDArray[np.float32],
    shape: tuple[int, int, int],
    order: int,
    color: tuple[float, float, float] | tuple[float, float, float, float],
) -> tuple[tuple[slice, ...], NDArray[np.float32] | None]:
    if len(color) == 4:
        _cr, _cg, _cb, _alpha = color
    else:
        _cr, _cg, _cb = color
        _alpha = 1.0
    sl_src, sl_dst = _prep_slices(start, stop, shape, img.shape)
    if sl_dst is None:
        return (slice(None),), None

    sl_dst = (slice(None),) + sl_dst

    transformed = affine_transform(
        img,
        mtx,
        mode="constant",
        cval=0.0,
        order=order,
        prefilter=False,
    )
    img_min = img.min()
    img_max = img.max()
    _a = (transformed[sl_src] - img_min) / (img_max - img_min) * _alpha
    color_array = np.stack([_cr * _a, _cg * _a, _cb * _a], axis=0)

    return sl_dst, color_array


def _simulate_2d_one(
    img: NDArray[np.float32],
    start: tuple[int, int, int],
    stop: tuple[int, int, int],
    mtx: NDArray[np.float32],
    shape: tuple[int, int, int],
    order: int,
):
    sl_src, sl_dst = _prep_slices(start, stop, shape, img.shape)
    if sl_dst is None:
        return (slice(None),), None
    transformed = affine_transform(
        img,
        mtx,
        mode="constant",
        cval=0.0,
        order=order,
        prefilter=False,
    )
    projected = np.sum(transformed[sl_src], axis=0)
    return sl_dst[1:], projected


def _simulate_projection_one(
    yx: NDArray[np.float32],
    shape: tuple[int, int],
    image: NDArray[np.float32],
    rotator: Rotation,
    glob_rotator: Rotation,
) -> tuple[tuple[slice, slice], NDArray[np.float32] | None]:
    proj_shape = np.array(image.shape[1:], dtype=np.int32)
    min_ = yx - proj_shape.astype(np.float32) / 2 + 0.5
    int_min_ = min_.astype(np.int32)
    int_max_ = int_min_ + proj_shape

    sl_src, sl_dst = _prep_slices(int_min_, int_max_, shape, proj_shape)

    if sl_dst is None:
        return (slice(None), slice(None)), None

    center = np.array(image.shape) / 2 - 0.5
    residue = np.array([0, *(min_ - int_min_)])
    mtx = _compose_affine_matrices(
        center, rotator.inv() * glob_rotator, center + residue
    )[0]

    transformed = affine_transform(
        image, mtx, mode="constant", cval=0.0, order=3, prefilter=False
    )
    projection: NDArray[np.float32] = np.sum(transformed, axis=0)
    return sl_dst, projection[sl_src]


def _simulate_projection_one_labeled(
    yx: NDArray[np.float32],
    shape: tuple[int, int],
    image: NDArray[np.float32],
    rotator: Rotation,
    glob_rotator: Rotation,
    idx: int,
) -> tuple[tuple[int, slice, slice], NDArray[np.float32] | None]:
    sl, img = _simulate_projection_one(yx, shape, image, rotator, glob_rotator)
    return (idx,) + sl, img


def cross(x: np.ndarray, y: np.ndarray, axis=None) -> np.ndarray:
    """Vector outer product in zyx coordinate."""
    return -np.cross(x, y, axis=axis)  # type: ignore
