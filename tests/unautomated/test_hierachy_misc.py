import pytest
import numpy as np
from models.hierarchy_misc import get_hierarchy_slices
from itertools import product
from functools import reduce

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

if __name__ == '__main__':
    test_default_usage(1)
    test_default_usage(2)
    test_default_usage(3)
    print('finish')
