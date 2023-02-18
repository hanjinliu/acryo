# acryo

An extensible cryo-EM/ET toolkit for Python.
The purpose of this library is to make data analysis on cryo-EM/ET more safe, efficient and reproducible.
Currently only the subtomogram averaging part is implemented.

[📘 Documentation](https://hanjinliu.github.io/acryo/)

### Install

```bash
pip install acryo -U
```

### Features

1. Out-of-core and parallel processing during subtomogram averaging/alignment to make full use of CPU.
2. Extensible and ready-to-use alignment models.
3. Manage subtomogram loading tasks from single or multiple tomograms in the same API.
4. Tomogram simulation.
5. Masked PCA clustering.
