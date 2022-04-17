# acryo

An extensible cryo-EM/ET toolkit for Python. Currently only the subtomogram averaging part is implemented.
The purpose of this library is to make data analysis on cryo-EM/ET more available for scientists.

### Highlights

1. Out-of-core and parallel processing using [dask](https://github.com/dask/dask).
   ![](images/task-graph.png)
2. Algorithms are highly extensible at low-level. You can customize pre-processing, alignment function and easily extend it to support multi-reference alignment and rotation optimization.
3. Concise representation of subtomograms, using `numpy.ndarray` for positions and `scipy.spatial.transform.Rotation` for orientation.
4. Tomogram generator for algorithm test.

### Contribution

Contribution is very welcome!
