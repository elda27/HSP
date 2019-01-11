import pytest
import numpy as np
from models.hierarchy_misc import get_hierarchy_slices, _binary_erosion, make_label_for_hierarchical_loss
from itertools import product
from functools import reduce
import cupy as cp

import mhd

def test_default_usage(n_level = 3):
    shape = (2 ** n_level,) * 3
    stack_shape = (2, 2, 2)
    a = np.zeros(shape)
    positions = [range(8) for i in range(n_level)]
    pos_iter = product(*positions)
    for pos in pos_iter:
        slices = get_hierarchy_slices(shape, pos, stack_shape=stack_shape)
        x = 0
        for i, p in enumerate(pos):
            x += (p + 1) * 10 ** i

        roi = a
        for s in slices:
            roi = roi[s]
        roi[0,0,0] = x

    mhd.write(f'tests/test_hierarchy_misc-simple_usage-level_{n_level}.mhd', a)

def test_erode():
    a = cp.ones((4, 1, 3,3,3), dtype=np.int32)
    a = cp.pad(a, ((0, 0), (0, 0), (2, 2), (2, 2), (2, 2)), 'constant', constant_values=0)
    b = _binary_erosion(a)

    mhd.write('tests/test_hierarchy_misc-erode-before.mhd', np.squeeze(cp.asnumpy(a[0])).astype(np.uint8))
    mhd.write('tests/test_hierarchy_misc-erode-after.mhd', np.squeeze(cp.asnumpy(b[0])).astype(np.uint8))

def test_make_label():
    a = cp.ones((4, 1, 3,3,3), dtype=np.int32)
    a = cp.pad(a, ((0,0), (0,0), (2,2), (2,2), (2,2)), 'constant', constant_values=0)
    label = make_label_for_hierarchical_loss(a)
    mhd.write('tests/test_hierarchy_misc-label-before.mhd', np.squeeze(cp.asnumpy(a[0])))
    mhd.write('tests/test_hierarchy_misc-label-after.mhd', np.squeeze(cp.asnumpy(label[0])))

if __name__ == '__main__':
    # test_default_usage(1)
    # test_default_usage(2)
    # test_default_usage(3)
    test_erode()
    test_make_label()
    print('finish')
