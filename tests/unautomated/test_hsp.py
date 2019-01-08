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
    def __init__(self):
        self.upsampled_counter = 1

    def __call__(self, x):
        data = []
        # for i in range(8):
        #     if x.data[0,0,0,0,0] == 1:
        #         self.upsampled_counter += 1
        #         n = self.upsampled_counter
        #     else:
        #         n = x.data[0, 0, 0, 0, 0]
        #     block = np.ones(x.shape, dtype=np.float32) * n * 8 + i
        #     data.append(
        #         chainer.Variable(block)
        #     )
        h = np.ones(
            x.shape[:2] + tuple(np.array(x.shape[2:]) * 2)
        )
        return h

class moc_CascadeOutputUnit(chainer.Chain):
    def __init__(self, out_ch):
        self.out_ch = out_ch
        self.counter = 1

    def __call__(self, x):
        self.counter += 1
        return x[:, :self.out_ch, ...] * self.counter


class moc_OutputUnit(chainer.Chain):
    def __init__(self, out_ch):
        self.out_ch = out_ch
        self.counter = 1

    def __call__(self, x):
        self.counter += 1
        return x[:, :self.out_ch, ...] * self.counter


def test_upsample(boundary_dicision, n_level):
    HierarchicalSurfacePredictor.Encoder = moc_Encoder
    HierarchicalSurfacePredictor.Decoder = moc_Decoder
    HierarchicalSurfacePredictor.CascadeOutputUnit = moc_CascadeOutputUnit
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