import pytest
import chainer
import mhd
import numpy as np
from models.concat_volume import concat_volume, ConcatVolume, concat_volume_slow


def test_concat():
    images = np.zeros((8, 8, 8), dtype=np.object)
    for i in range(images.size):
        index = np.unravel_index(i, images.shape)
        images[index] = chainer.Variable(np.ones((1, 1, 2, 2, 2)) * i)

    v = concat_volume(images.ravel().tolist(), stack_shape=images.shape, pad=0)
    mhd.write('tests/test_concat.mhd', np.squeeze(v.data))

def test_concat_about_reference():
    images = np.zeros((8, 8, 8), dtype=np.object)
    for i in range(images.size):
        index = np.unravel_index(i, images.shape)
        images[index] = chainer.Variable(np.ones((1, 1, 2, 2, 2)) * i)

    v1 = concat_volume_slow(images.ravel().tolist(), stack_shape=images.shape, pad=0)
    v2 = concat_volume(images.ravel().tolist(), stack_shape=images.shape, pad=0)

    assert np.all(v1.data == v2.data)

def test_concat_with_pad():
    shape = (8, 8, 8)
    images = np.zeros(shape, dtype=np.object)
    images_with_pad = np.zeros(shape, dtype=np.object)
    for i in range(images.size):
        index = np.unravel_index(i, images.shape)
        source = np.ones((1, 1, 2, 2, 2)) * i
        with_pad = np.pad(source, ((0, 0), (0, 0), (2, 2), (2, 2), (2, 2)), 'constant', constant_values=images.size * 2)
        images[index] = chainer.Variable(source)
        images_with_pad[index] = chainer.Variable(with_pad)

    source = concat_volume(images.ravel().tolist(), stack_shape=shape, pad=0)
    with_pad = concat_volume(images_with_pad.ravel().tolist(), stack_shape=shape, pad=0)
    without_pad = concat_volume(images_with_pad.ravel().tolist(), stack_shape=shape, pad=2)

    assert np.all(source.data == without_pad.data)

if __name__ == '__main__':
    test_concat_about_reference()
    # test_concat()
    # test_concat_with_pad()