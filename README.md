[![Python package index download statistics](https://img.shields.io/pypi/dm/acryo.svg)](https://pypistats.org/packages/acryo)
[![PyPI version](https://badge.fury.io/py/acryo.svg)](https://badge.fury.io/py/acryo)

# acryo

`acryo` is an extensible cryo-EM/ET toolkit for Python.

The purpose of this library is to make data analysis of cryo-EM/ET safer, efficient, reproducible and customizable for everyone.
Scientists can avoid the error-prone CLI-based data handling, such as writing out the results to the files every time and manage all the result just by the file names.

[📘 Documentation](https://hanjinliu.github.io/acryo/)

### Install

###### Use pip

```bash
pip install acryo -U
```

###### From source

```bash
git clone git+https://github.com/hanjinliu/acryo.git
cd acryo
pip install -e .
```

### Features

1. Out-of-core and parallel processing during subtomogram averaging/alignment to make full use of CPU.
2. Extensible and ready-to-use alignment models.
3. Manage subtomogram loading tasks from single or multiple tomograms in the same API.
4. Tomogram and tilt series simulation.
5. Masked PCA clustering.

### Code Snippet

```Python
import polars as pl
from acryo import SubtomogramLoader, Molecules  # acryo objects
from acryo.pipe import soft_otsu  # data input pipelines

# construct a loader
loader = SubtomogramLoader.imread(
    "path/to/tomogram.mrc",
    molecules=Molecules.from_csv("path/to/molecules.csv"),
)

# filter out bad alignment in polars way
loader_filt = loader.filter(pl.col("score") > 0.7)

# averaging
avg = loader_filt.average(output_shape=(48, 48, 48))

# alignment
aligned_loader = loader.align(
    template=avg,                       # use the average as template
    mask=soft_otsu(sigma=2, radius=2),  # apply soft-Otsu to template to make the mask
    tilt_range=(-50, 50),               # range of tilt series degrees.
    cutoff=0.5,                         # lowpass filtering cutoff
    max_shifts=(4, 4, 4),               # search space limits
)

```
