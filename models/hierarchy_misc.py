import numpy as np
import cupy as cp
from scipy.ndimage import binary_erosion
from chainer.backends import cuda
from chainer import functions as F

def get_hierarchy_slices(shape, pos, stride=1, stack_shape=(2, 2, 2)):
    slices = []
    block_shape = np.array(shape)
    stack_shape = np.array(stack_shape)
    for level, p in enumerate(pos):
        index = np.unravel_index(p, stack_shape)
        strides = (stride * 2 ** (len(pos) - level - 1),) * 3

        index = tuple(i * s for i, s in zip(index, strides))

        block_shape = block_shape // stack_shape

        slices.append(
            tuple(slice(i, i + s) for i, s in zip(index, block_shape))
        )
    return slices


def _binary_erosion(a):
    if cuda.get_array_module(a) == np:
        return binary_erosion(a)

    if a.dtype.kind != 'f':
        a = a.astype(np.float32)

    neighbor = cp.ones((1, 1, 3, 3, 3), dtype=np.float32)
    count = F.convolution_nd(a, neighbor, pad=1).data
    return count > 26.0

def make_label_for_hierarchical_loss(label):
    xp = cuda.get_array_module(label)
    indices = xp.unique(label)
    ndim = indices.size * 2 - 1

    output = xp.zeros_like(label, dtype=xp.int32)
    for i in indices[1:]:
        mask = label == i
        occupied_mask = _binary_erosion(mask)
        output[mask] = int(i * 2 - 1)
        output[occupied_mask] = int(i * 2)
    return output