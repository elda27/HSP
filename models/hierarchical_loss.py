import chainer
import chainer.functions as F
from chainer import reporter
import cupy as cp

from models.hierarchy_misc import make_label_for_hierarchical_loss

class HierarchicalLoss(chainer.Chain):
    def __init__(self, model, keep_inference=False, observer=None):
        self.keep_inference = keep_inference
        self.observer = self if observer is None else observer
        super().__init__(model=model)

    def forward(self, x, t):
        hierarchical_losses = {}

        y, hy = self.model(x, save_hierarchy=True)
        loss_final = F.softmax_cross_entropy(y, t)
        loss =  len(hy) * loss_final

        # Save current inference
        if self.keep_inference:
            self.y = y
            self.hy = hy

        hy = sorted(hy.items(), key=lambda x: x[0], reverse=True)
        dt = F.expand_dims(chainer.Variable(t.astype(cp.float32)), axis=1)
        for level, volume in hy:
            dt = F.max_pooling_3d(dt, ksize=2, stride=2)
            downsampled_label = make_label_for_hierarchical_loss(dt.data)
            hl = F.softmax_cross_entropy(
                volume,
                chainer.Variable(downsampled_label[:, 0, ...])
            )
            hierarchical_losses[level] = hl
            loss += hl

        # Report losses
        metrics = {'loss_l{}'.format(k):v for k, v in hierarchical_losses.items()}
        metrics['loss_l{}'.format(max(hierarchical_losses) + 1)] = loss_final
        metrics['loss'] = loss
        reporter.report(metrics, self.observer)

        return loss

