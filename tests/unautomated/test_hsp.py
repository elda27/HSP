import pytest
import chainer
import chainer.functions as F
from models.hsp import HierarchicalSurfacePredictor
from models.concat_volume import concat_volume
import numpy as np
import mhd
from scipy.signal import resample


class moc_Encoder(chainer.Chain):
    def __init__(self, in_ch, out_ch):
        self.in_ch = in_ch
        self.out_ch = out_ch

    def __call__(self, x):
        return x


class moc_Decoder(chainer.Chain):
    def __init__(self, out_shape=(16, 16, 16)):
        self.out_shape = out_shape

    def __call__(self, x):
        out_shape = x.shape[:2] + self.out_shape
        out = np.ones(out_shape)
        return out


class moc_UpsampleUnit(chainer.Chain):
    instance_count=0
    def __init__(self):
        moc_UpsampleUnit.instance_count += 1
        self.counter = moc_UpsampleUnit.instance_count
        self.upsampled_counter = 1

    def __call__(self, x):
        data = []
        self.upsampled_counter += 1
        p = np.amax(x.data)
        digit = int(np.log10(p) + 1)
        h = p * np.ones(
            x.shape[:2] + tuple(np.array(x.shape[2:]) * 2 - 4)
        ) + 100 ** (self.counter) * self.upsampled_counter
        return h

class moc_OutputUnit(chainer.Chain):
    def __init__(self, out_ch):
        self.out_ch = out_ch
        self.counter = 1

    def __call__(self, x):
        self.counter += 1
        return x[:, :self.out_ch, ...] + self.counter

@pytest.mark.parametrize("boundary_dicision,n_level", [
    ('boundary', 4),
    ('unboundary', 4),
    ('random', 4),
    ('boundary', 5),
    ('unboundary', 5),
    ('random', 5),
])
def test_upsample(boundary_dicision, n_level):
    HierarchicalSurfacePredictor.Encoder = moc_Encoder
    HierarchicalSurfacePredictor.Decoder = moc_Decoder
    HierarchicalSurfacePredictor.CascadeOutputUnit = moc_OutputUnit
    HierarchicalSurfacePredictor.UpsampleUnit = moc_UpsampleUnit
    HierarchicalSurfacePredictor.OutputUnit = moc_OutputUnit

    HierarchicalSurfacePredictor.always_boudary_dicision = boundary_dicision

    feature = chainer.Variable(np.ones((1, 32, 20, 20, 20), dtype=np.float32))

    model = HierarchicalSurfacePredictor(out_ch=3, n_level=n_level)
    hierarchy_volumes = {
        i: model.get_hierarchy_output_array(i)
        for i in range(1, model.n_level)
    }
    volume = model.upsample_cascade(feature, hierarchy_volumes)
    hierarchy_volumes = model.concat_hierarchy_volumes(hierarchy_volumes)

    for ch in range(3):
        mhd.write('tests/{}_test_hsp_always_{}_volume_ch_{}.mhd'.format(n_level, boundary_dicision, ch), volume.data[0, ch])
        for level, v in hierarchy_volumes.items():
            mhd.write('tests/{}_test_hsp_always_{}_volume_ch_{}_level_{}.mhd'.format(n_level, boundary_dicision, ch, level), v.data[0, ch])

if __name__ == '__main__':
    test_upsample('boundary', 4)
    test_upsample('unboundary', 4)
    test_upsample('random', 4)
    test_upsample('boundary', 5)
    test_upsample('unboundary', 5)
    test_upsample('random', 5)
    print('finish!')