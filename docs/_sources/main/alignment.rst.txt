===============
Alignment Model
===============

An alignment model defines the protocols for pre-transformation and alignment scoring.

.. contents:: Contents
    :local:
    :depth: 1

Alignment Workflows
===================

There are three types of alignment workflows.

Single template
---------------

If the alignment model is created with a single template image, the workflow is very simple.
It masks and pre-transforms both the sub-volume and the template, and then aligns them.

.. mermaid::

    graph LR
        vol(sub-volume\nat <i>i</i>-th molecule)
        tmp[[template image]]
        vol_t(transformed\nsub volume)
        tmp_t[[transformed\ntemplate]]
        aln{alignment}
        result[alignment results]

        vol--masking &<br>pre-transformation-->vol_t-->aln
        tmp--masking &<br>pre-transformation-->tmp_t-->aln
        aln-->result


Multiple templates
------------------

If the alignment model is created with multiple template images, masking, pre-transformation
and alignment are performed for each template separately. The alignment result with the best
score will be considered as the optimal result.

.. mermaid::

    graph LR

        subgraph Subvolumes
            vol(sub-volume\nat <i>i</i>-th molecule)
            vol_t(transformed\nsub volume)
            vol--masking &<br>pre-transformation-->vol_t
        end

        subgraph Templates
            tmp0[[template image A]]
            tmp1[[template image B]]
            tmp2[[template image C]]
            tmp0_t[[transformed\ntemplate A]]
            tmp1_t[[transformed\ntemplate B]]
            tmp2_t[[transformed\ntemplate C]]
            tmp0--masking &<br>pre-transformation-->tmp0_t
            tmp1--masking &<br>pre-transformation-->tmp1_t
            tmp2--masking &<br>pre-transformation-->tmp2_t
        end

        subgraph Alignment
            aln0{alignment}
            aln1{alignment}
            aln2{alignment}
        end
        result[best alignment results]

        vol_t-->aln0
        vol_t-->aln1
        vol_t-->aln2

        tmp0_t-->aln0
        tmp1_t-->aln1
        tmp2_t-->aln2

        aln0-->result
        aln1-->result
        aln2-->result

Single template with rotation
-----------------------------

Many alignment methods do not search for the optimal rotation of the template image. In this
case, rotated templates will be generated and used for alignment. Essentially, it is the same
as the multiple-template workflow.

.. mermaid::

    graph LR

        subgraph Subvolumes
            vol(sub-volume\nat <i>i</i>-th molecule)
            vol_t(transformed\nsub volume)
            vol--masking &<br>pre-transformation-->vol_t
        end

        subgraph Templates
            tmp[[template image]]
            tmp_t[[transformed\ntemplate]]
            tmp--masking &<br>pre-transformation-->tmp_t

            tmp0[[template image A]]
            tmp1[[template image B]]
            tmp2[[template image C]]
            rot{image rotation}
            tmp_t-->rot
            rot-->tmp0
            rot-->tmp1
            rot-->tmp2
        end

        subgraph Alignment
            aln0{alignment}
            aln1{alignment}
            aln2{alignment}
        end
        result[best alignment results]

        vol_t-->aln0
        vol_t-->aln1
        vol_t-->aln2

        tmp0-->aln0
        tmp1-->aln1
        tmp2-->aln2

        aln0-->result
        aln1-->result
        aln2-->result

Ready-to-use Models
===================

There are now two alignment models that can be used directly.

1. :class:`ZNCCAlignment`
    Model that align subvolumes using ZNCC (Zero-mean Normalized Cross Correlation) score.
2. :class:`PCCAlignment`
    Model that align subvolumes using PCC (Phase Cross Correlation) score.

Both models are implemented with low-pass filtering, template rotation and missing
wedge masking.

Model construction
------------------

.. code-block:: python

    from acryo.alignment import ZNCCAlignment

    model = ZNCCAlignment(
        template,  # template image
        mask,      # mask image
        rotations=[(10, 5), (4, 2), (8, 4)],
        cutoff=0.5,
        tilt_range=(-60, 60),
    )

- Shape of ``mask`` must be the same as ``template``. ``template * mask`` and
  ``subvolume * mask`` will be used for alignment.

- ``rotations`` can be three tuples or a :class:`scipy.spatial.transform.Rotation` object.

  - If three tuples are given, each tuple defines the maximum rotation angle and the increment
    around z, y or x (external) axis. The unit is degree. For example, the first ``(10, 5)``
    means that the rotation angles -10, -5, 0, 5, 10 will be used for the rotation around z axis.

  - If a :class:`scipy.spatial.transform.Rotation` object is given, all the rotations in the
    object will be used for alignment. Make sure that the identity rotation is included.

- ``cutoff`` is the relative cutoff frequency for low-pass filtering. The Nyquist frequency is
  :math:`0.5 \times \sqrt{3} = 0.866` for 3D images.

- ``tilt_range`` is the range of tilt series angles in degree.

Align images
------------

The :meth:`align` method is used to align a sub-volume to the template image of the model.
Note that this method does not actually transform the sub-volume to the template. It only
calculate the optimal shift/rotation parameters. To transform the sub-volume, use :meth:`fit`.

.. code-block:: python

    result = model.align(
        subvolume,
        max_shifts,
        quaternion,
        pos,
    )

- ``subvolume`` is the sub-volume to be aligned. It must be a 3D array with the same shape
  as the template.
- ``max_shifts`` is a tuple of maximum shifts in z, y and x direction. The unit is pixel but
  it can be a float number.
- ``quaternion`` is the rotation of the sub-volume in the original tomogram. It must be a (4,)
  :class:`numpy.ndarray` object of quaternion. If you are using :class:`acryo.Molecules`,
  its quaternions can directly be used here. This is basically used to mask the missing wedge.
- ``pos`` is the position of the sub-volume in the original tomogram. It must be a (3,)
  :class:`numpy.ndarray` object. Default alignment models does not use this parameter.

The return value ``result`` is a named-tuple :class:`AlignmentResult` object. It contains the
following fields.

.. code-block:: python

    class AlignmentResult(NamedTuple):
        label: int
        shift: NDArray[np.float32]
        quat: NDArray[np.float32]
        score: float

- ``label`` is the integer label of the best alignment if multiple templates are used.
- ``shift`` is the optimal shift in z, y and x direction.
- ``quat`` is the optimal rotation in quaternion.
- ``score`` is the alignment score of the best alignment.

Fit images
----------

The :meth:`fit` method is used to transform the sub-volume to fit the template image of the
model. It is essentially the same as calling :meth:`align` for every rotation and then
Affine transform the sub-volume to the best alignment result, but :meth:`fit` is faster
because it parallelizes the rotation and alignment processes.

.. code-block:: python

    result = model.fit(
        subvolume,
        max_shifts,
        cval=0.0,
    )

- ``subvolume`` and ``max_shifts`` is the same as :meth:`align`.
- ``cval`` is the constant value used for Affine transformations. 1% percentile will be used
  by default.

Correlation landscape
---------------------

The word "correlation landscape" came from "energy landscape" in the context of protein
folding. It is a 3D array of the correlation scores between the sub-volume and the template
image.

.. code-block:: python

    arr = model.landscape(
        subvolume,
        max_shifts,
    )

- ``subvolume`` is the sub-volume to be aligned. It must be a 3D array with the same shape
  as the template.
- ``max_shifts`` is a tuple of maximum shifts in z, y and x direction. The unit is pixel but
  it can be a float number.
- ``quaternion`` is the rotation of the sub-volume in the original tomogram. It must be a (4,)
  :class:`numpy.ndarray` object of quaternion. If you are using :class:`acryo.Molecules`,
  its quaternions can directly be used here. This is basically used to mask the missing wedge.
- ``pos`` is the position of the sub-volume in the original tomogram. It must be a (3,)
  :class:`numpy.ndarray` object. Default alignment models does not use this parameter.

Define Custom Alignment Model
=============================

In :mod:`acryo.alignment`, there are several abstract base classes that can be used to
efficiently define custom alignment models.

- :class:`BaseAlignmentModel` ... The most basic one that provides the minimum interface.
  Need to override :meth:`_optimize` and :meth:`pre_transform`.
- :class:`RotationImplemented` ... Rotated templates will be generated even if the
  optimization algorithm does not optimize the rotation. Need to override :meth:`_optimize`
  and :meth:`pre_transform`.
- :class:`TomographyInput` ... Rotation, low-pass filtering and missing wedge masking is
  already implemented. Only need to override :meth:`_optimize`.

When you override methods, the following should be noted.

- :meth:`pre_transform`

    This method must have the following signature.

    .. code-block:: python

        def pre_transform(self, image: NDArray[np.float32]) -> NDArray[np.complex64]:
            ...

    The input image could be either the sub-volume or the template image. It is masked by
    the input mask image but is not masked by the missing wedge mask in :class:`TomographyInput`.
    The output image will be directly passed to the :meth:`_optimize` method, so the data
    type depends on the implementation.

- :meth:`_optimize`

    This method must have the following signature.

    .. code-block:: python

        def _optimize(
            self,
            subvolume: NDArray[T],
            template: NDArray[T],
            max_shifts: tuple[float, float, float],
            quaternion: NDArray[np.float32],
            pos: NDArray[np.float32],
        ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
            ...

    This method is called for every set of sub-volume and template images.

    - ``subvolume`` and ``template`` is the images *after* pre-transformation.
      Thus, they could be Fourier transformed.
    - ``max_shift`` is directly passed from :meth:`align` or :meth:`fit` method.
    - ``quaternion`` is the rotation of the sub-volume. This parameter can be used
      to mask the missing wedge.
    - ``pos`` is the position of the sub-volume in the original tomogram. Its
      unit is pixel. This parameter can be used for CTF correction of defocusing.
    - The return value must be a tuple of ``(shift, rotation, score)``.

      - ``shift`` is the optimal shift in z, y and x direction. More precisely,
        ``ndi.shift(img, -shift)`` will properly align the image to the template.
      - ``rotation`` is the optimal rotation in quaternion. If the alignment model
        does not optimize the rotation, this value should be ``array([0, 0, 0, 1])``.
      - ``score`` is the score of the alignment. Larger score means better alignment.
