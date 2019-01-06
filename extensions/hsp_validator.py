import copy

from itertools import product
from pathlib import Path

import chainer
from chainer import configuration
from chainer.dataset import convert
from chainer import functions as F
from chainer import reporter as reporter_module
from chainer.training import extension
from chainer.backends import cuda

import mhd
from models import HierarchicalLoss

class HspValidator(extension.Extension):

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    name = None

    def __init__(
        self, iterator, target, device=-1,
        n_save=3, n_eval=None, converter=None
    ):
        self.target = target

        self.iterator = iterator
        self.device = device

        self.n_save = n_save
        self.n_eval = n_eval

        self.converter = converter if convert is not None else self._converter

    def _converter(self, batch):
        return list(zip(*batch))

    def _to_gpu(self, xs):
        xp = cuda.get_array_module(xs[0])
        x = chainer.Variable(
            xp.concatenate([x[xp.newaxis, ...] for x in xs], axis=0)
        )
        x.to_gpu(self.device)
        return x.data

    def _to_cpu(self, x, unwrapped=True):
        cpu_x = F.copy(x, -1)
        if unwrapped:
            return cpu_x.data
        else:
            return cpu_x

    def get_iterator(self, name):
        """Returns the iterator of the given name."""
        return self._iterators[name]

    def get_target(self, name):
        """Returns the target link of the given name."""
        return self._targets[name]

    def __call__(self, trainer=None):
        """Executes the validator extension.
        Unlike usual extensions, this extension can be executed without passing
        a trainer object. This extension reports the performance on validation
        dataset using the :func:`~chainer.report` function. Thus, users can use
        this extension independently from any trainer by manually configuring
        a :class:`~chainer.Reporter` object.
        Args:
            trainer (~chainer.training.Trainer): Trainer object that invokes
                this extension. It can be omitted in case of calling this
                extension manually.
        Returns:
            dict: Result dictionary that contains mean statistics of values
            reported by the evaluation function.
        """
        # set up a reporter
        reporter = reporter_module.Reporter()
        if self.name is not None:
            prefix = self.name + '/'
        else:
            prefix = 'validation/'
            #prefix = ''

        name = 'main'
        reporter.add_observer(prefix + name, self.target)

        with reporter:
            with configuration.using_config('train', False):
                result = self.validate(trainer.out, trainer.updater.iteration)

        reporter_module.report(result)
        return result

    def validate(self, output_dir, iteration):
        output_dir = Path(output_dir)
        iterator = self.iterator

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        n_valid = 0

        for batch in it:
            observation = {}
            xs, ts = self._converter(batch)
            projections = self._to_gpu(xs)
            labels = self._to_gpu(ts)

            model = HierarchicalLoss(
                self.target,
                keep_inference=True,
                observer=self.target
            )

            with reporter_module.report_scope(observation):
                model(projections, labels)

            # Save images
            if n_valid < self.n_save:
                self.save_images(output_dir, iteration, model, xs, ts)

            n_valid += len(batch)
            if self.n_eval is not None and n_valid > self.n_eval:
                break

            summary.add(observation)

        return summary.compute_mean()

    def save_images(self, output_dir, iteration, model, xs, ts):
        pred_label_probs = model.y
        pred_hierarchy_label_probs = model.hy

        pred_labels = F.argmax(F.softmax(pred_label_probs), axis=1)
        cpu_pred_labels = self._to_cpu(pred_labels, unwrapped=True)
        for i in range(cpu_pred_labels.shape[0]):
            for name, image in zip(['pred_label', 'label', 'projection'], [cpu_pred_labels, ts, xs]):
                filename = 'valid_{:08d}_{:03d}_{}.mhd'.format(iteration, i, name)
                mhd.write(str(output_dir / filename), image[i])

        for level, image in pred_hierarchy_label_probs.items():
            h_pred_labels = F.argmax(F.softmax(image), axis=1)
            cpu_h_pred_labels = self._to_cpu(h_pred_labels, unwrapped=True)
            for i in range(cpu_h_pred_labels.shape[0]):
                filename = 'valid_{:08d}_{:03d}_pred_label_level_{}.mhd'.format(iteration, i, level)
                mhd.write(
                    str(output_dir / filename),
                    cpu_h_pred_labels[i]
                )