import random
from functools import reduce

import numpy as np

import chainer
import chainer.functions as F
from chainer.backends import cuda

from models.encoder import Encoder
from models.decoder import Decoder
from models.upsample_unit import UpsampleUnit
from models.output_unit import OutputUnit
from models.cascade_output_unit import CascadeOutputUnit

from models.concat_volume import concat_volume


class HierarchicalSurfacePredictor(chainer.Chain):
    """
    Hierarchical volume reconstruction from 2d image.
    A detail is see the following paper:
        C. Hane, et. al., “Hierarchical surface prediction for 3D object reconstruction,”
        Proc. - 2017 Int. Conf. 3D Vision, 3DV 2017, pp. 412–420, 2018.
    """

    def __init__(
        self,
        in_ch=1, out_ch=1, n_level=5,
        block_size=(16, 16, 16),
        encoder=None, upsample_threshold=0.1,
        upsample_prob_rule=None
    ):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        if out_ch == 1:
            self.cascade_out_ch = 3
        else:
            self.cascade_out_ch = 2 * out_ch - 1
        self.n_level = n_level - 1 # zero-based level
        self.block_size = block_size
        self.out_size = tuple(b * (2 ** n_level) for b in block_size)
        self.upsample_threshold = upsample_threshold
        self.upsample_prob = 1e-3

        with self.init_scope():
            if encoder is not None:
                self.encoder = encoder
            else:
                self.encoder = Encoder(
                    in_ch, out_ch
                )
            self.decoder = Decoder()
            setattr(self, 'O%02d' % 0, CascadeOutputUnit(
                out_ch=self.cascade_out_ch
            ))
            for i in range(1, self.n_level):
                setattr(self, 'O%02d' % i, CascadeOutputUnit(
                    out_ch=self.cascade_out_ch
                ))
                setattr(self, 'U%02d' % i, UpsampleUnit())
            setattr(self, 'O%02d' % self.n_level, OutputUnit(
                out_ch=out_ch
            ))

    def is_upsample(self, cascade):
        if chainer.config.train and random.random() > self.upsample_prob:
            return True
        else:
            n_elem = reduce(lambda x, y: x*y, cascade.shape[1:])
            return float(F.max(cascade).data) < self.upsample_threshold

    def compute_slices(self, index, size, pad=2, num_block=(2, 2, 2)):
        slices = []
        index = np.unravel_index(index, num_block)
        for j, s, nb in zip(index, size, num_block):
            slices.append(slice(
                j * s,
                (j + 1) * s + pad * 2
            ))
        return tuple(slices)

    def __call__(self, x, save_hierarchy=False):
        if chainer.config.train:
            self.upsample_prob += 1e-4
        xp = cuda.get_array_module(x)
        latent_vector = self.encoder(x)
        F = self.decoder(latent_vector)

        hierarchy_volumes = {} if save_hierarchy else None
        output_volume = self.upsample_cascade(F, hierarchy_volumes)

        if save_hierarchy:
            output_hierarchy_volumes = {}
            for k, v in hierarchy_volumes.items():
                if len(v) == 1:
                    output_hierarchy_volumes[k] = v[0]
                else:
                    output_hierarchy_volumes[k] = concat_volume(
                        v, pad=0
                    )
            output_hierarchy_volumes[0] = self.get_cascade_output(
                self.O00(F)
            )[:, :, 2:18, 2:18, 2:18]

            return output_volume, output_hierarchy_volumes
        else:
            return output_volume

    def upsample_cascade(self, f, hierarchy_volumes=None, level=1, pos=None):

        def get_item(x, region, channel=None):
            if channel is None:
                return x[(slice(x.shape[0]), slice(x.shape[1])) + region]
            else:
                return x[(slice(x.shape[0]), channel) + region]

        upsampled_feature = None
        pred_volume = getattr(self, 'O%02d' % level)(f)

        cascade_outputs = []
        hierarchy_outputs = []
        for i in range(8):
            in_slices = self.compute_slices(i, self.block_size)

            if level != self.n_level:
                is_boundary = level == 1 or any(self.is_upsample(
                    get_item(pred_volume, in_slices, channel=j)
                ) for j in range(1, pred_volume.shape[1], 2))
                if is_boundary:
                    if upsampled_feature is None:
                        upsampled_feature = getattr(self, 'U%02d' % level)(f)
                    if hierarchy_volumes is not None:
                        hierarchy_outputs.append(
                            self.get_cascade_output(pred_volume)
                        )
                    cascade_output = self.upsample_cascade(
                        get_item(upsampled_feature, in_slices),
                        hierarchy_volumes,
                        level=level+1
                    )
                else:
                    cascade_output = F.unpooling_3d(
                        self.get_cascade_output(pred_volume),
                        ksize=2, stride=2, cover_all=False
                    )
                    if hierarchy_volumes is not None:
                        hierarchy_outputs.append(cascade_output)

                cascade_outputs.append(cascade_output)
            else:  # not to upsample on the final level.
                cascade_outputs.append(pred_volume)
                # hierarchy_outputs.append(cascade_output)

        if self.n_level == level:
            pad = 2
        else:
            pad = 0

        if hierarchy_volumes is not None and level != self.n_level:
            hierarchy_volumes.setdefault(level, []).append(
                concat_volume(hierarchy_outputs, pad=2)
            )

        return concat_volume(cascade_outputs, pad=pad)

    def get_cascade_output(self, pred_volume):
        volumes = (pred_volume[:, 0, ],) + tuple(
            pred_volume[:, i, ...] + pred_volume[:, i + 1, ...]
            for i in range(1, pred_volume.shape[1], 2)
        )
        volumes = tuple(F.expand_dims(x, axis=1) for x in volumes)
        return F.concat(volumes, axis=1)
