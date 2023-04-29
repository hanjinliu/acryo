# pyright: reportPrivateImportUsage=false

from __future__ import annotations
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import (
    Callable,
    Iterable,
    NamedTuple,
    Sequence,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import Self
import warnings

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from acryo._rotation import normalize_rotations
from acryo._types import Ranges, subpixel, degree
from acryo._utils import (
    compose_matrices,
    missing_wedge_mask,
    lowpass_filter_ft,
)
from acryo._typed_scipy import ifftn, spline_filter, affine_transform, map_coordinates
from acryo._dask import DaskTaskPool
from ._bound import ParametrizedModel

if TYPE_CHECKING:
    pass


TemplateType = Union[NDArray[np.float32], Sequence[NDArray[np.float32]]]
MaskType = Union[
    NDArray[np.float32],
    Callable[[NDArray[np.float32]], NDArray[np.float32]],
    None,
]
AlignmentFactory = Callable[
    [TemplateType, Union[NDArray[np.float32], None]], "BaseAlignmentModel"
]


class AlignmentResult(NamedTuple):
    """The optimal alignment result."""

    label: int
    shift: NDArray[np.float32]
    quat: NDArray[np.float32]
    score: float

    def affine_matrix(self, shape: tuple[int, int, int]) -> NDArray[np.float32]:
        """Return the affine matrix."""
        rotator = Rotation.from_quat(self.quat)
        shift_matrix = np.eye(4, dtype=np.float32)
        shift_matrix[:3, 3] = self.shift
        rot_matrix = compose_matrices(np.array(shape) / 2 - 0.5, [rotator])[0]
        return shift_matrix @ rot_matrix


class BaseAlignmentModel(ABC):
    """
    The base class to implement alignment method.

    This class supports subvolume masking, pre-transformation of subvolumes and
    template, optimization of spatial transformation.

    subvolume    template
        |            |
        v            v
       ( soft-masking )
        |            |
        v            v
      (pre-transformation)
        |            |
        +-----+------+
              |
              v
         (alignment)

    Abstract Methods
    ----------------
    >>> def optimize(self, template, reference, max_shifts, quaternion):
    >>>     ...
    >>> def pre_transform(self, image):
    >>>     ...

    """

    _DUMMY_POS = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    _DUMMY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def __init__(
        self,
        template: TemplateType,
        mask: MaskType = None,
    ):
        if isinstance(template, np.ndarray):
            if template.dtype != np.float32:
                template = template.astype(np.float32)
            self._template = template
            self._n_templates = 1
            self._ndim = template.ndim
        else:
            self._template = np.stack(template, axis=0)
            if self._template.dtype != np.float32:
                self._template: NDArray[np.float32] = self._template.astype(np.float32)
            self._n_templates = self._template.shape[0]
            self._ndim = self._template.ndim - 1

        if callable(mask):
            if self._n_templates != 1:
                raise ValueError("Cannot create a mask using multiple templates")
            self._mask: NDArray[np.float32] = mask(self._template)
            if self._template.shape != self._mask.shape:
                raise ValueError(
                    "Shape mismatch in between template image "
                    f"{self._template.shape} and mask image {self._mask.shape})."
                )
        elif mask is None:
            self._mask = np.ones(self._template.shape[-self._ndim :], dtype=np.float32)
        else:
            if self._template.shape[-self._ndim :] != mask.shape:
                raise ValueError(
                    "Shape mismatch in between template image "
                    f"{self._template.shape[-self._ndim :]} and mask image {mask.shape})."
                )
            if mask.dtype not in (np.float32, np.bool_):
                mask = mask.astype(np.float32)
            self._mask = mask

        self._template_input, self._mask_input = self._get_template_and_mask_input()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self._template.shape})"

    @property
    def template(self) -> NDArray[np.float32]:
        """The template image."""
        return self._template

    @property
    def mask(self) -> NDArray[np.float32]:
        """The mask image."""
        return self._mask

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Return the array shape of input images and template."""
        return self._template.shape[-self._ndim :]

    @classmethod
    def with_params(
        cls,
        **params,
    ):
        """Create a BaseAlignmentModel instance with parameters."""
        return ParametrizedModel(cls, **params)

    @abstractmethod
    def _optimize(
        self,
        subvolume: NDArray[np.complex64],
        template: NDArray[np.complex64],
        max_shifts: tuple[float, ...],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """
        Optimization of shift and rotation of subvolume.

        This method uses a subvolume and a template image to find the optimal
        shift/rotation to fit subvolume to template under the restriction of
        maximum shifts.

        Parameters
        ----------
        subvolume : np.ndarray
            Input subvolume. This array has the same dimensionality as the template.
        template : np.ndarray
            Template image.
        max_shifts : tuple of float
            Maximum shifts in each axis. This value should be float pixel. If
            optimization requires integer input, it is better to convert this
            parameter to tuple of integers.

        Returns
        -------
        (3,) np.ndarray, (4,) np.ndarray and float
            Local shift, local rotation and score. "Local" means you don't have
            to consider the position and orientation of molecules.
            - shift ... Shift in pixel.
            - rotation ... Rotation in quaternion. If this cannot be optimized by the
              implemented algorithm, this value can be ``[0., 0., 0., 0.]``.
            - score ... This value Must be a float and larger value should represent
              better results.
        """

    @abstractmethod
    def pre_transform(self, image: NDArray[np.float32]) -> NDArray[np.complex64]:
        """Pre-transformation applied to input images (including template)."""

    def _landscape(
        self,
        subvolume: NDArray[np.complex64],
        template: NDArray[np.complex64],
        max_shifts: tuple[float, ...],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Return the landscape of the subvolume."""
        raise NotImplementedError(
            f"_landscape method is not implemented for {type(self).__name__}."
        )

    def _get_template_and_mask_input(
        self,
    ) -> tuple[NDArray[np.complex64], NDArray[np.float32]]:
        """
        Returns proper template and mask images for alignment.

        Template dimensionality will be dispatched according to the input
        parameters. Returned template should be used in line of the
        :func:`get_alignment_function`.

        Returns
        -------
        array
            Template image(s) and mask image(s). Its axes varies depending on the input.

            - single template image ... 3D
            - many template images ... 4D
        """
        if self._n_templates > 1:
            template_input = np.stack(
                [self.pre_transform(tmp * self._mask) for tmp in self._template],
                axis=0,
            )
        else:
            template_input = self.pre_transform(self._template * self._mask)
        return template_input, self._mask

    def align(
        self,
        img: NDArray[np.float32],
        max_shifts: tuple[float, float, float],
        quaternion: NDArray[np.float32] | None = None,
        pos: NDArray[np.float32] | None = None,
    ) -> AlignmentResult:
        """
        Align an image using current alignment parameters.

        Parameters
        ----------
        img : np.ndarray
            Subvolume to be aligned
        max_shifts : tuple[float, float, float]
            Maximum shifts along z, y, x axis in pixel.

        Returns
        -------
        AlignmentResult
            Result of alignment.
        """
        if quaternion is None:
            _quat = self._DUMMY_QUAT
        else:
            _quat = quaternion
        if pos is None:
            _pos = self._DUMMY_POS
        else:
            _pos = pos

        if self._is_multiple():
            _align_fn = self._optimize_multiple
        else:
            _align_fn = self._optimize_single
        return _align_fn(
            img,
            self._template_input,
            self._mask_input,
            max_shifts,
            _quat,
            _pos,
        )

    def fit(
        self,
        img: NDArray[np.float32],
        max_shifts: tuple[float, float, float],
        cval: float | None = None,
    ) -> tuple[NDArray[np.float32], AlignmentResult]:
        """
        Fit image to template based on the alignment model.

        Parameters
        ----------
        img : np.ndarray
            Input image that will be transformed.
        max_shifts : tuple[float, float, float]
            Maximum shifts along z, y, x axis in pixel.
        cval : float, optional
            Constant value for padding.

        Returns
        -------
        np.ndarray, AlignmentResult
            Transformed input image and the alignment result.
        """
        result = self.align(img, max_shifts=max_shifts, quaternion=None, pos=None)
        mtx = result.affine_matrix(img.shape)
        _cval = _normalize_cval(cval, img)
        img_trans = affine_transform(img, mtx, cval=_cval)
        return img_trans, result

    def landscape(
        self,
        img: NDArray[np.float32],
        max_shifts: tuple[float, float, float],
        quaternion: NDArray[np.float32] | None = None,
        pos: NDArray[np.float32] | None = None,
        upsample: int = 1,
    ) -> NDArray[np.float32]:
        """
        Calculate correlation landscape of the input image.

        Parameters
        ----------
        img : np.ndarray
            Subvolume for the landscape calculation.
        max_shifts : tuple[float, float, float]
            Maximum shifts along z, y, x axis in pixel.

        Returns
        -------
        np.ndarray
            N (if single template) or N+1 (if multi-template) dimensional array of
            correlation landscape.
        """
        if quaternion is None:
            _quat = self._DUMMY_QUAT
        else:
            _quat = quaternion
        if pos is None:
            _pos = self._DUMMY_POS
        else:
            _pos = pos
        if self._is_multiple():
            fn = self._landscape_multiple
        else:
            fn = self._landscape_single

        # calculate the landscape
        lds = fn(
            img,
            self._template_input,
            self._mask_input,
            max_shifts,
            _quat,
            _pos,
        )

        if upsample > 1:
            if not self._is_multiple():
                coords = _create_mesh_for_landscape(lds.shape, max_shifts, upsample)
                lds_upsampled = map_coordinates(
                    lds, coords, order=3, mode="constant", cval=0.0, prefilter=True
                )
            else:
                coords = _create_mesh_for_landscape(lds.shape[1:], max_shifts, upsample)
                all_lds = [
                    map_coordinates(
                        l, coords, order=3, mode="constant", cval=0.0, prefilter=True
                    )
                    for l in lds
                ]
                lds_upsampled = np.stack(all_lds, axis=0)
            return lds_upsampled
        return lds

    def _landscape_single(
        self,
        subvolume: NDArray[np.float32],
        template: NDArray[np.complex64],
        mask: NDArray[np.float32],
        max_shifts: tuple[float, ...],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
    ):
        return self._landscape(
            self.pre_transform(subvolume * mask),
            template,
            max_shifts=max_shifts,
            quaternion=quaternion,
            pos=pos,
        )

    def _landscape_multiple(
        self,
        subvolume: NDArray[np.float32],
        template_list: Iterable[NDArray[np.complex64]],
        mask_list: Iterable[NDArray[np.float32]],
        max_shifts: tuple[float, float, float],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
    ):
        out: list[NDArray[np.float32]] = []
        for template, mask in zip(template_list, mask_list):
            lnd = self._landscape(
                self.pre_transform(subvolume * mask),
                template,
                max_shifts=max_shifts,
                quaternion=quaternion,
                pos=pos,
            )
            out.append(lnd)
        return np.stack(out, axis=0)

    def _optimize_single(
        self,
        subvolume: NDArray[np.float32],
        template: NDArray[np.complex64],
        mask: NDArray[np.float32],
        max_shifts: tuple[float, float, float],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
    ) -> AlignmentResult:
        out = self._optimize(
            self.pre_transform(subvolume * mask),
            template,
            max_shifts=max_shifts,
            quaternion=quaternion,
            pos=pos,
        )
        return AlignmentResult(0, *out)

    def _optimize_multiple(
        self,
        subvolume: NDArray[np.float32],
        template_list: Iterable[NDArray[np.complex64]],
        mask_list: Iterable[NDArray[np.float32]],
        max_shifts: tuple[float, float, float],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
    ) -> AlignmentResult:
        all_shifts: list[np.ndarray] = []
        all_quat: list[np.ndarray] = []
        all_score: list[float] = []
        for template, mask in zip(template_list, mask_list):
            shift, quat, score = self._optimize(
                self.pre_transform(subvolume * mask),
                template,
                max_shifts=max_shifts,
                quaternion=quaternion,
                pos=pos,
            )
            all_shifts.append(shift)
            all_quat.append(quat)
            all_score.append(score)

        iopt = int(np.argmax(all_score))
        return AlignmentResult(iopt, all_shifts[iopt], all_quat[iopt], all_score[iopt])

    def _is_multiple(self) -> bool:
        return self.niter > 1

    @property
    def is_multi_templates(self) -> bool:
        """
        Whether alignment parameters requires multi-templates.
        "Multi-template" includes alignment with subvolume rotation.
        """
        return self._is_multiple()

    @property
    def niter(self) -> int:
        """Number of templates."""
        return self._n_templates


# deprecated
def optimize(self: BaseAlignmentModel, *args, **kwargs):
    warnings.warn(
        "`optimize` is deprecated. It is now a private method.", DeprecationWarning
    )
    return self._optimize(*args, **kwargs)


class RotationImplemented(BaseAlignmentModel):
    """
    An alignment model implemented with default rotation optimizer.

    If ``optimize`` does not support rotation optimization, this class implements
    simple parameter searching algorithm to it. Thus, ``optimize`` only has to
    optimize shift of images.
    """

    def __init__(
        self,
        template: TemplateType,
        mask: MaskType = None,
        rotations: Ranges | None = None,
    ):
        self.quaternions = normalize_rotations(rotations)
        self._n_rotations = self.quaternions.shape[0]
        super().__init__(template=template, mask=mask)

    @classmethod
    def with_params(
        cls,
        *,
        rotations: Ranges | None = None,
    ) -> ParametrizedModel[Self]:
        """Create an alignment model instance with parameters."""
        return ParametrizedModel(cls, rotations=rotations)

    @property
    def has_rotation(self) -> bool:
        """If the alignment model has rotation optimization."""
        return self._n_rotations > 1

    def align(
        self,
        img: NDArray[np.float32],
        max_shifts: tuple[subpixel, subpixel, subpixel],
        quaternion: NDArray[np.float32] | None,
        pos: NDArray[np.float32] | None,
    ) -> AlignmentResult:
        """
        Align an image using current alignment parameters.

        Parameters
        ----------
        img : np.ndarray
            Subvolume to be aligned
        max_shifts : tuple of float
            Maximum shifts along z, y, x axis in pixel.

        Returns
        -------
        AlignmentResult
            Result of alignment.
        """
        iopt, shift, _, corr = super().align(img, max_shifts, quaternion, pos)
        quat = self.quaternions[iopt % self._n_rotations]
        return AlignmentResult(label=iopt, shift=shift, quat=quat, score=corr)

    def fit(
        self,
        img: NDArray[np.float32],
        max_shifts: tuple[subpixel, subpixel, subpixel],
        cval: float | None = None,
    ) -> tuple[NDArray[np.float32], AlignmentResult]:
        """
        Fit image to template based on the alignment model.

        Unlike ``BaseAlignmentModel``, rotation optimization is executed in
        parallel to boost calculation.

        Parameters
        ----------
        img : np.ndarray
            Input image that will be transformed.
        max_shifts : tuple of float
            Maximum shifts along z, y, x axis in pixel.
        cval : float, optional
            Constant value for padding.

        Returns
        -------
        np.ndarray, AlignmentResult
            Transformed input image and the alignment result.
        """
        pool = DaskTaskPool.from_func(self._optimize)
        pos = np.zeros(3, dtype=np.float32)
        for quat, tmp, mask in zip(
            self.quaternions, self._template_input, self._mask_input
        ):
            pool.add_task(
                self.pre_transform(img * mask),
                tmp,
                max_shifts,
                quat,
                pos=pos,
            )
        results = pool.compute()
        scores = [x[2] for x in results]
        iopt = np.argmax(scores)
        opt_result = results[iopt]
        result = AlignmentResult(
            label=0,
            shift=opt_result[0],
            quat=self.quaternions[iopt],
            score=opt_result[2],
        )

        mtx = result.affine_matrix(img.shape)
        _img_cval = _normalize_cval(cval, img)
        img_trans = affine_transform(img, mtx, cval=_img_cval)
        return img_trans, result

    def _transform_template(
        self,
        temp: NDArray[np.float32],
        matrix: NDArray[np.float32],
        cval: float | None = None,
        order: int = 3,
        prefilter: bool = True,
    ) -> NDArray[np.complex64]:
        _cval = _normalize_cval(cval, temp)

        return self.pre_transform(
            affine_transform(
                temp, matrix=matrix, cval=_cval, order=order, prefilter=prefilter
            )
        )

    def _get_template_and_mask_input(
        self,
    ) -> tuple[NDArray[np.complex64], NDArray[np.float32]]:
        """
        Returns proper template image for alignment.

        Template dimensionality will be dispatched according to the input
        parameters. Returned template should be used in line of the
        :func:`get_alignment_function`.

        Returns
        -------
        np.ndarray
            Template image(s). Its axes varies depending on the input.

            - no rotation, single template image ... 3D
            - has rotation, single template image ... 4D
            - no rotation, many template images ... 4D
            - has rotation, many template images ... 4D and when iterated over
              the first axis yielded images will be (rot0, temp0),
              (rot0, temp1), ...
        """
        if self._n_rotations > 1:
            rotators = [Rotation.from_quat(r).inv() for r in self.quaternions]
            matrices = compose_matrices(
                np.array(self._template.shape[-3:]) / 2 - 0.5, rotators
            )
            cval = float(np.percentile(self._template, 1))
            if self._n_templates > 1:
                # all_templates: list[da.Array] = []
                # all_masks: list[da.Array] = []
                inputs_templates: list[NDArray[np.float32]] = [
                    spline_filter(
                        tmp * self._mask,
                        order=3,
                        mode="constant",
                        output=np.float32,
                    )
                    for tmp in self._template
                ]
                pool_template = DaskTaskPool.from_func(self._transform_template)
                pool_mask = DaskTaskPool.from_func(affine_transform)
                ntmp = len(inputs_templates)
                for mat in matrices:
                    for tmp in inputs_templates:
                        pool_template.add_task(
                            tmp, mat, order=3, cval=cval, prefilter=False
                        )
                        # all_templates.append(
                        #     da.from_delayed(
                        #         self._transform_template_delayed(
                        #             tmp, mat, order=3, cval=cval, prefilter=False
                        #         ),
                        #         shape=tmp.shape,
                        #         dtype=tmp.dtype,
                        #     )
                        # )
                    pool_mask.add_tasks(
                        ntmp, self._mask, mat, order=3, mode="nearest", prefilter=False
                    )

                    # all_masks.extend(
                    #     [
                    #         delayed_affine(
                    #             self._mask,
                    #             mat,
                    #             order=3,
                    #             mode="nearest",
                    #             prefilter=False,
                    #         )
                    #     ]
                    #     * len(inputs_templates)
                    # )
                template_input = np.stack(pool_template.compute(), axis=0)
                mask_input = np.stack(pool_mask.compute(), axis=0)
                # template_input, mask_input = da.compute(
                #     da.stack(all_templates, axis=0), da.stack(all_masks, axis=0)
                # )

            else:
                template_masked = spline_filter(
                    self._template * self._mask,
                    order=3,
                    output=np.float32,
                    mode="constant",
                )
                pool_template = DaskTaskPool.from_func(self._transform_template)
                pool_mask = DaskTaskPool.from_func(affine_transform)
                for mat in matrices:
                    pool_template.add_task(
                        template_masked, mat, order=3, cval=cval, prefilter=False
                    )
                    pool_mask.add_task(
                        self._mask, mat, order=3, mode="nearest", prefilter=False
                    )
                template_input = np.stack(pool_template.compute(), axis=0)
                mask_input = np.stack(pool_mask.compute(), axis=0)
        else:
            pool = DaskTaskPool.from_func(self.pre_transform)
            if self._n_templates > 1:
                for tmp in self._template:
                    pool.add_task(tmp * self._mask)
                template_input = np.stack(pool.compute(), axis=0)
                mask_input = np.stack([self._mask] * len(self._template), axis=0)

            else:
                # NOTE: dask.compute is always called once inside this method.
                template_input = pool.add_task(self._template * self._mask).compute()[0]
                mask_input = self._mask

        return template_input, mask_input

    def _is_multiple(self) -> bool:
        return self._n_templates * self._n_rotations > 1

    @property
    def niter(self) -> int:
        """Number of iteration per sub-volume."""
        return self._n_templates * self._n_rotations


class TomographyInput(RotationImplemented):
    """
    An alignment model that implements missing-wedge masking and low-pass filter.

    This alignment model is useful for subtomogram averaging of real experimental
    data with limited tilt ranges. Template image will be masked with synthetic
    missing-wedge mask in the frequency domain.
    """

    def __init__(
        self,
        template: TemplateType,
        mask: MaskType = None,
        rotations: Ranges | None = None,
        cutoff: float | None = None,
        tilt_range: tuple[degree, degree] | None = None,
    ):
        self._cutoff = cutoff or 1.0
        if tilt_range is not None:
            deg0, deg1 = tilt_range
            if deg0 >= deg1:
                raise ValueError("Tilt range must be in form of (min, max).")
        self._tilt_range = tilt_range
        super().__init__(template, mask, rotations)

    @classmethod
    def with_params(
        cls,
        *,
        rotations: Ranges | None = None,
        cutoff: float | None = None,
        tilt_range: tuple[degree, degree] | None = None,
    ) -> ParametrizedModel[Self]:
        """Create an alignment model instance with parameters."""
        return ParametrizedModel(
            cls,
            rotations=rotations,
            cutoff=cutoff,
            tilt_range=tilt_range,
        )

    def pre_transform(self, image: NDArray[np.float32]) -> NDArray[np.complex64]:
        """Apply low-pass filter without IFFT."""
        return lowpass_filter_ft(image, cutoff=self._cutoff)

    def masked_difference(
        self,
        image: NDArray[np.float32],
        quaternion: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """
        Difference between an image and the template, considering the missing wedge.

        Parameters
        ----------
        image : 3D array
            Input image, usually a subvolume from a tomogram.
        quaternion : (4,) array
            Rotation of the image, usually the quaternion array of a Molecules
            object.

        Returns
        -------
        3D array
            Difference map.
        """
        if self._n_templates > 1:
            raise NotImplementedError(
                "Masked difference is not implemented for multi-template."
            )
        ft = self._template_input  # NOTE: ft.ndim == 3
        ft[:] = self.mask_missing_wedge(ft, quaternion)
        template_masked = np.real(ifftn(ft))
        img_input = np.real(ifftn(self.pre_transform(image * self._mask)))
        return img_input - template_masked

    def mask_missing_wedge(
        self,
        image: NDArray[np.complex64],
        quaternion: NDArray[np.float32],
    ) -> NDArray[np.complex64]:
        """Apply missing wedge mask in the frequency domain."""
        return image * self._get_missing_wedge_mask(quaternion)

    def _get_missing_wedge_mask(
        self, quat: NDArray[np.float32]
    ) -> NDArray[np.float32] | int:
        """
        Create a binary mask that covers tomographical missing wedge.

        Parameters
        ----------
        quat : (4,) array
            Quaternion representation of the orientation of the subvolume.

        Returns
        -------
        np.ndarray or float
            Missing wedge mask. If ``tilt_range`` is None, 1 will be returned.
        """
        if self._tilt_range is None:
            return 1
        return missing_wedge_mask(
            rotator=Rotation.from_quat(quat),
            tilt_range=self._tilt_range,
            shape=self.input_shape,
        )


def _normalize_cval(cval: float | None, img: np.ndarray) -> float:
    if cval is None:
        _cval = float(np.percentile(img, 1))
    else:
        _cval = cval
    return _cval


@lru_cache(maxsize=2)
def _create_mesh_for_landscape(
    shape: tuple[int, int, int],
    max_shifts: tuple[float, float, float],
    upsample: int,
) -> NDArray[np.float32]:
    upsampled_max_shifts = (np.asarray(max_shifts) * upsample).astype(np.int32)
    center = np.array(shape) / 2 - 0.5
    mesh = np.meshgrid(
        *[
            np.arange(-width, width + 1) / upsample + c
            for c, width in zip(center, upsampled_max_shifts)
        ],
        indexing="ij",
    )
    return np.stack(mesh, axis=0)
