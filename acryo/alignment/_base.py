from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, NamedTuple, Sequence
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation
from dask import array as da, delayed

from ._utils import normalize_rotations, lowpass_filter, lowpass_filter_ft
from .._types import Ranges, subpixel
from .._utils import compose_matrices


class AlignmentResult(NamedTuple):
    """The optimal alignment result."""

    label: int
    shift: np.ndarray
    quat: np.ndarray
    score: float


class BaseAlignmentModel(ABC):
    """
    The base class to implement alignment method.

    Abstract Methods
    ----------------
    >>> def optimize(self, subvolume, reference, max_shifts, quaternion):
    >>>     ...
    >>> def pre_transform(self, image):
    >>>     ...

    """

    def __init__(
        self,
        template: np.ndarray | Sequence[np.ndarray],
        mask: np.ndarray | None = None,
    ):
        if isinstance(template, np.ndarray):
            self._template = template
            self._n_templates = 1
            self._ndim = template.ndim
        else:
            self._template: np.ndarray = np.stack(template, axis=0)
            self._n_templates = self._template.shape[0]
            self._ndim = self._template.ndim - 1

        if mask is None:
            self.mask = 1
        else:
            if self._template.shape[-self._ndim :] != mask.shape:
                raise ValueError(
                    "Shape mismatch in zyx axes between tempalte image "
                    f"{self._template.shape} and mask image {mask.shape})."
                )
            self.mask = mask

        self._align_func = self._get_alignment_function()
        self._template_input: np.ndarray | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self._template.shape})"

    @property
    def template_input(self) -> np.ndarray:
        if self._template_input is None:
            self._template_input = self._get_template_input()
        return self._template_input

    @abstractmethod
    def optimize(
        self,
        subvolume: np.ndarray,
        template: np.ndarray,
        max_shifts: tuple[float, ...],
        quaternion: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
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
    def pre_transform(self, image: np.ndarray) -> np.ndarray:
        """Pre-transformation applied to input images (including template)."""

    def _get_template_input(self) -> np.ndarray:
        """
        Returns proper template image for alignment.

        Template dimensionality will be dispatched according to the input
        parameters. Returned template should be used in line of the
        :func:`get_alignment_function`.

        Returns
        -------
        ip.ImgArray
            Template image(s). Its axes varies depending on the input.

            - single template image ... 3D
            - many template images ... 4D
        """
        if self.is_multi_templates:
            template_input = np.stack(
                [self.pre_transform(tmp * self.mask) for tmp in self._template],
                axis=0,
            )
        else:
            template_input = self.pre_transform(self._template)
        return template_input

    def align(
        self,
        img: np.ndarray,
        max_shifts: tuple[float, float, float],
        quaternion: np.ndarray | None = None,
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
        img_masked = img * self.mask
        if quaternion is None:
            _quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            _quat = quaternion
        return self._align_func(
            self.pre_transform(img_masked),
            self.template_input,
            max_shifts,
            _quat,
        )

    def fit(
        self,
        img: np.ndarray,
        max_shifts: tuple[float, float, float],
        cval: float | None = None,
    ) -> tuple[np.ndarray, AlignmentResult]:
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
        result = self.align(img, max_shifts=max_shifts, quaternion=None)
        rotator = Rotation.from_quat(result.quat)
        matrix = compose_matrices(np.array(img.shape) / 2 - 0.5, [rotator])[0]
        if cval is None:
            _cval = np.percentile(img, 1)
        else:
            _cval = cval
        img_shifted = ndi.shift(img, -result.shift, cval=_cval)
        img_trans = ndi.affine_transform(img_shifted, matrix, cval=_cval)
        return img_trans, result

    def _optimize_single(
        self,
        subvolume: np.ndarray,
        template: np.ndarray,
        max_shifts: tuple[float, float, float],
        quaternion: np.ndarray,
    ) -> AlignmentResult:
        out = self.optimize(subvolume, template, max_shifts, quaternion)
        return AlignmentResult(0, *out)

    def _optimize_multiple(
        self,
        subvolume: np.ndarray,
        template_list: Iterable[np.ndarray],
        max_shifts: tuple[float, float, float],
        quaternion: np.ndarray,
    ) -> AlignmentResult:
        all_shifts: list[np.ndarray] = []
        all_quat: list[np.ndarray] = []
        all_score: list[float] = []
        for template in template_list:
            shift, quat, score = self.optimize(
                subvolume,
                template,
                max_shifts,
                quaternion,
            )
            all_shifts.append(shift)
            all_quat.append(quat)
            all_score.append(score)

        iopt = int(np.argmax(all_score))
        return AlignmentResult(iopt, all_shifts[iopt], all_quat[iopt], all_score[iopt])

    def _get_alignment_function(self):
        if self.is_multi_templates:
            return self._optimize_multiple
        else:
            return self._optimize_single

    @property
    def is_multi_templates(self) -> bool:
        """
        Whether alignment parameters requires multi-templates.
        "Multi-template" includes alignment with subvolume rotation.
        """
        return self._n_templates > 1


class RotationImplemented(BaseAlignmentModel):
    """
    An alignment model implemented with default rotation optimizer.

    If ``optimize`` does not support rotation optimization, this class implements
    simple parameter searching algorithm to it. Thus, ``optimize`` only has to
    optimize shift of images.
    """

    def __init__(
        self,
        template: np.ndarray | Sequence[np.ndarray],
        mask: np.ndarray | None = None,
        rotations: Ranges | None = None,
    ):
        self.quaternions = normalize_rotations(rotations)
        self._n_rotations = self.quaternions.shape[0]
        super().__init__(template=template, mask=mask)

    def align(
        self,
        img: np.ndarray,
        max_shifts: tuple[subpixel, subpixel, subpixel],
        quaternion: np.ndarray | None,
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
        iopt, shift, _, corr = super().align(img, max_shifts, quaternion)
        quat = self.quaternions[iopt % self._n_rotations]
        return AlignmentResult(label=iopt, shift=shift, quat=quat, score=corr)

    def fit(
        self,
        img: np.ndarray,
        max_shifts: tuple[subpixel, subpixel, subpixel],
        cval: float | None = None,
    ) -> tuple[np.ndarray, AlignmentResult]:
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
        img_masked = img * self.mask
        delayed_optimize = delayed(self.optimize)
        delayed_transform = delayed(self._transform_template)
        template_masked = self._template * self.mask
        cval = np.percentile(self._template, 1)
        rotators = [Rotation.from_quat(r).inv() for r in self.quaternions]
        matrices = compose_matrices(
            np.array(self._template.shape[-self._ndim :]) / 2 - 0.5, rotators
        )
        tasks = []
        for mat in matrices:
            tmp = delayed_transform(template_masked, mat, cval=cval)
            task = delayed_optimize(
                img_masked, tmp, max_shifts, np.array([0.0, 0.0, 0.0, 1.0])
            )
            tasks.append(task)
        results: list[tuple] = da.compute(tasks)[0]
        scores = [x[2] for x in results]
        iopt = np.argmax(scores)
        opt_result = results[iopt]
        result = AlignmentResult(
            0, opt_result[0], self.quaternions[iopt], opt_result[2]
        )

        rotator = Rotation.from_quat(result.quat)
        matrix = compose_matrices(np.array(img.shape) / 2 - 0.5, [rotator])[0]
        if cval is None:
            _cval = np.percentile(img, 1)
        else:
            _cval = cval
        img_shifted = ndi.shift(img, -result.shift, cval=_cval)
        img_trans = ndi.affine_transform(img_shifted, matrix, cval=_cval)
        return img_trans, result

    @delayed
    def _delayed_optimize(
        self,
        subvol: np.ndarray,
        template_list: Iterable[np.ndarray],
        max_shifts: tuple[subpixel, subpixel, subpixel],
        quaternion: np.ndarray,
    ) -> AlignmentResult:
        all_shifts: list[np.ndarray] = []
        all_quat: list[np.ndarray] = []
        all_score: list[float] = []
        for template in template_list:
            shift, quat, score = self.optimize(
                subvol,
                template,
                max_shifts,
                quaternion,
            )
            all_shifts.append(shift)
            all_quat.append(quat)
            all_score.append(score)

        iopt = int(np.argmax(all_score))
        return AlignmentResult(iopt, all_shifts[iopt], all_quat[iopt], all_score[iopt])

    def _transform_template(
        self,
        temp: np.ndarray,
        matrix: np.ndarray,
        cval: float | None = None,
    ) -> np.ndarray:
        if cval is None:
            _cval = np.percentile(temp, 1)
        else:
            _cval = cval

        return self.pre_transform(ndi.affine_transform(temp, matrix=matrix, cval=_cval))

    def _get_template_input(self) -> np.ndarray:
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
            cval = np.percentile(self._template, 1)
            if self.is_multi_templates:
                all_templates: list[np.ndarray] = []
                for mat in matrices:
                    for tmp in self._template:
                        all_templates.append(
                            self._transform_template(tmp * self.mask, mat)
                        )

                template_input: np.ndarray = np.stack(all_templates, axis=0)

            else:
                template_masked = self._template * self.mask
                template_input: np.ndarray = np.stack(
                    [
                        self._transform_template(template_masked, mat, cval=cval)
                        for mat in matrices
                    ],
                    axis=0,
                )
        else:
            if self.is_multi_templates:
                template_input: np.ndarray = np.stack(
                    [self.pre_transform(tmp * self.mask) for tmp in self._template],
                    axis=0,
                )
            else:
                template_input = self.pre_transform(self._template)

        return template_input

    def _get_alignment_function(self):
        if self.is_multi_templates or self._n_rotations > 1:
            return self._optimize_multiple
        else:
            return self._optimize_single


class FrequencyCutoffImplemented(RotationImplemented):
    """
    An alignment model that supports frequency-based pre-filtering.

    This class can be used for implementing such as low-pass filter or high-
    pass filter before alignment.
    """

    def __init__(
        self,
        template: np.ndarray | Sequence[np.ndarray],
        mask: np.ndarray | None = None,
        rotations: Ranges | None = None,
        cutoff: float | None = None,
    ):
        self._cutoff = cutoff or 1.0
        super().__init__(template, mask, rotations)


class FourierLowpassInput(FrequencyCutoffImplemented):
    """Abstract model that uses low-pass-filtrated Fourier images as inputs."""

    def pre_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply low-pass filter without IFFT."""
        return lowpass_filter_ft(image, cutoff=self._cutoff)


class RealLowpassInput(FrequencyCutoffImplemented):
    """Abstract model that uses low-pass-filtrated real images as inputs."""

    def pre_transform(self, image: np.ndarray) -> np.ndarray:
        """Apply low-pass filter."""
        return lowpass_filter(image, cutoff=self._cutoff)
