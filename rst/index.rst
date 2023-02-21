.. acryo documentation master file, created by
   sphinx-quickstart on Wed Feb 15 19:45:14 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

acryo
=====

:mod:`acryo` is an extensible cryo-EM/ET toolkit for Python.

`Jump to GitHub <https://github.com/hanjinliu/acryo>`_

.. toctree::
   :maxdepth: 1
   :caption: Contents:

Installation
============

.. code-block:: bash

   pip install acryo -U

Before getting start with :mod:`acryo`, it is highly recommended to understand the concept of
out-of-core computation using `dask <https://docs.dask.org/en/stable/array.html>`_, 3D rotation using
`scipy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html>`_,
and table data processing using `polars <https://pola-rs.github.io/polars-book/user-guide/>`_.

Contents
========

.. toctree::
   :maxdepth: 1

   ./main/molecules
   ./main/loader
   ./main/caching
   ./main/pipe
   ./main/alignment


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
