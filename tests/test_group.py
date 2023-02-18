from acryo import SubtomogramLoader, Molecules
import numpy as np
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_group():
    img = np.zeros((30, 30, 35), dtype=np.float32)
    mole = Molecules(
        pos=[[15, 15, 10 + i] for i in range(15)], features={"x": np.arange(15) % 3}
    )

    loader = SubtomogramLoader(img, mole, output_shape=(9, 9, 9))
    return loader.groupby("x")


def test_average():
    group = _get_group()
    group.average()
    group.average((11, 11, 11))


def test_average_split():
    group = _get_group()
    group.average_split()
    group.average_split(n_set=3, output_shape=(11, 11, 11))


def test_align():
    group = _get_group()
    template = np.ones((9, 9, 9), dtype=np.float32)
    out = group.align(template)


def test_align_no_template():
    group = _get_group()
    out = group.align_no_template()


def test_align_multi_templates():
    group = _get_group()
    template1 = np.ones((9, 9, 9), dtype=np.float32)
    template2 = np.ones((9, 9, 9), dtype=np.float32) * 2
    out = group.align_multi_templates([template1, template2])
