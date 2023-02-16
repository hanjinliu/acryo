==================
Subtomogram Loader
==================

A :class:`SubtomogramLoader` is a pair of a 3D tomogram image and a
:class:`Molecules` object, with some additional parameters.

.. code-block:: python

    def __init__(self, image, molecules, order=3, scale=1.0, output_shape=Unset(), corner_safe=False): ...

1. ``image`` (`numpy.ndarray` or `dask.Array`) ... the tomogram image.
2. ``molecules`` (`Molecules`) ... molecules in the tomogram.
3. ``order`` (`int`) ... order of the spline interpolation. 0=nearest, 1=linear,
    3=cubic.
4. ``scale`` (`float`) ... scale (physical/pixel) of the tomogram image. This
    parameter must match the positions of ``molecules``.
5. ``output_shape`` (`tuple`) ... shape of the output subtomograms, which will be
    used to determine the subtomogram shape during subtomogram averaging.
6. ``corner_safe`` (`bool`) ... if true, the subtomogram loader will ensure that
    the volume inside the given output shape will not be affected after rotation,
    otherwise the corners of the subtomograms will be dimmer.

Subtomogram averaging
=====================

:meth:`SubtomogramLoader.average` crops all the subtomograms around the molecules and
average them. This method always returns a 3D :class:`numpy.ndarray` object.

.. code-block:: python

    from dask import array as da
    from acryo import SubtomogramLoader, Molecules

    image = da.random.random((100, 100, 100))
    molecules = Molecules([[40, 40, 60], [60, 60, 40]])

    # give output shape beforehand
    loader = SubtomogramLoader(image, molecules, output_shape=(64, 64, 64))
    avg = loader.average()

    # or give output shape after construction
    loader = SubtomogramLoader(image, molecules)
    avg = loader.average(output_shape=(64, 64, 64))

Subtomogram Alignment
=====================

:meth:`SubtomogramLoader.align` crops all the subtomograms around the molecules and
align them to the given template image (reference image). This method will return
a new :class:`SubtomogramLoader` object with the updated :class:`Molecules` object.

You have to provide a template image, optionally a mask image and an alignment model.

.. code-block:: python

    from dask import array as da
    from acryo import SubtomogramLoader, Molecules
    from acryo.alignment import ZNCCAlignment

    image = da.random.random((100, 100, 100))
    template = np.random.random((20, 20, 20))
    molecules = Molecules([[40, 40, 60], [60, 60, 40]])

    loader = SubtomogramLoader(image, molecules)
    out = loader.align(template, alignment_model=ZNCCAlignment)

If you want to give parameters to the alignment model, you can use the :meth:`with_params`
method of alignment model classes. It returns a factory function for the parametrized model.

.. code-block:: python

    loader = SubtomogramLoader(image, molecules)
    out = loader.align(template, alignment_model=ZNCCAlignment.with_params(cutoff=0.5))
