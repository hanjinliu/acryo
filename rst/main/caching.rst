==================
Cache Subtomograms
==================

Loading subtomograms from a tomogram is a computationally expensive operation; The proper
regions of the tomogram must be read from disk, and will be shifted/rotated to the proper
position/orientation.

Nevertheless, sometimes you'll have to load subtomograms from the same places many times.
A typical example is when you want to create a template image by subtomogram averaging,
and use the template to align molecules.

.. code-block:: python

    from acryo import SubtomogramLoader, Molecules
    loader = SubtomogramLoader.imread(
        "/path/to/tomogram.mrc",
        molecules=Molecules.from_csv("/path/to/molecules.csv"),
        output_shape=(50, 50, 50),
    )

    template = loader.average()  # create template
    aligned = loader.align(template)  # align molecules to template

.. note::

    Of course, this simple example is what :meth:`align_no_template` does in more efficient
    way.

In this example, same set of subtomograms is loaded twice.

Create Cache of Loading Tasks
=============================

Subtomogram loaders have :meth:`cached` context manager. Within this context, subtomograms
of the given shape will temporarily be saved in a file, and will be loaded from there if
possible.

.. code-block:: python

    with loader.cached():  # take a while to create cache
        template = loader.average()  # much faster
        aligned = loader.align(template)  # much faster

Cache Inheritance
=================

:meth:`filter` and :meth:`groupby` inherits the cache of the parent loader. For instance,
if you want to create a template from the well-aligned molecules, you can do the following:

.. code-block:: python

    with loader.cached():  # take a while to create cache
        loader_filt = loader.filter(pl.col("score") > 0.7)
        template = loader_filt.average()  # much faster
        aligned = loader.align(template)  # much faster

Here, ``loader_filt.average()`` requires a subset of subtomograms that are already cached by
``loader.cached()``, which is also available from ``loader_filt.average()``.
