import argparse
import os
import numpy as np
import random

import matplotlib

from models import HierarchicalSurfacePredictor, HierarchicalLoss

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers
from chainerui.utils import save_args
from chainer.iterators import MultiprocessIterator, SerialIterator
from chainer.backends import cuda

from datasets import get_crossval_patients, Dataset
from extensions import CGANValidator, LogReport
from extensions import HspValidator
import utils
from chainer.training.extensions import LinearShift

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    cuda.cupy.random.seed(seed)


def main():
    parser = argparse.ArgumentParser(
        description='chainer implementation of pix2pix')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=100,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--frequency', type=int, default=-1,
                        help='Frequency of taking a snapshot')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--dataset', required=True,
                        help='Directory of image files.')
    # parser.add_argument('--cvlist', required=True)
    # parser.add_argument('--cvindex', type=int, default=0)
    parser.add_argument('--train-index', type=str, required=True)
    parser.add_argument('--valid-index', type=str, required=True)
    parser.add_argument('--exp-name', type=str, default='reconstruction')
    parser.add_argument('--out', '-o', default='logs',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--n-level', type=int, default=3)
    args = parser.parse_args()

    set_random_seed(args.seed)

    # setup output directory
    # timestamp = get_vcs_timestamp()
    out = utils.get_new_training_log_dir(os.path.join(
        args.out, args.exp_name
    ))
    os.makedirs(out, exist_ok=True)
    save_args(args, out)

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('# Log directory: ' + out)
    print('')

    # setup models
    model = HierarchicalSurfacePredictor(out_ch=3, n_level=args.n_level)
    opt_model = HierarchicalLoss(model)

    if args.gpu >= 0:
        # cuda.get_device(args.gpu).use()
        cuda.get_device_from_id(args.gpu).use()
        model.to_gpu(args.gpu)

    # setup optimizers
    def make_optimizer(model, alpha=0.0002, beta1=0.5):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001), 'hook_dec')
        return optimizer

    optimizer = make_optimizer(opt_model)

    # setup dataset
    train_patients = get_patient_index(args.train_index)
    valid_patients = get_patient_index(args.valid_index)

    train = Dataset(
        args.dataset, train_patients)
    valid = Dataset(
        args.dataset, valid_patients)
    # test = Dataset(args.dataset, patients['test'])

    if args.debug:
        iter_type = SerialIterator
    else:
        iter_type = MultiprocessIterator

    train_iter = iter_type(train, args.batchsize,
                           repeat=True, shuffle=True)
    valid_iter = iter_type(valid, args.batchsize,
                           repeat=False, shuffle=True)
    # test_iter = MultiprocessIterator(test,  args.batchsize,
    #                                  repeat=False, shuffle=False)

    print('# train: {}'.format(len(train)))
    print('# valid: {}'.format(len(valid)))
    # print('# test : {}'.format(len(test)))
    print('')

    # setup a trainer
    updater = chainer.training.StandardUpdater(
        iterator=train_iter,
        optimizer=optimizer,
        device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=out)

    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(extensions.snapshot(
        filename='snapshot_iter_{.updater.iteration:08}.npz'),
        trigger=(frequency, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'hsp_iter_{.updater.iteration:08}.npz'), trigger=(frequency, 'epoch'))

    train_loss_layers = ['main/loss_l{}'.format(i) for i in range(model.n_level + 1)]
    valid_loss_layers = ['validation/main/loss_l{}'.format(i) for i in range(model.n_level + 1)]
    log_keys = ['main/loss',
                'validation/main/loss'] + train_loss_layers + valid_loss_layers

    trainer.extend(LogReport(keys=log_keys, trigger=(100, 'iteration')))

    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss'] + train_loss_layers,
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(['validation/main/loss'] + valid_loss_layers,
                                  'epoch', file_name='valid.png'))

    trainer.extend(extensions.PrintReport(['epoch', 'iteration'] + log_keys))

    trainer.extend(extensions.ProgressBar())

    # setup a validator
    validator = HspValidator(
        iterator=valid_iter,
        target=model,
        device=args.gpu,
        n_eval=1000)
    trainer.extend(
        validator,
        trigger=(2500, 'iteration'))
    validator(trainer)

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # run
    trainer.run()

    # setup a inferener
    # infer = Inferencer(test_iter, gen, device=args.gpu)
    # y = infer.run()
    #
    # # save files
    # label_files = test.label_files
    # common_path = os.path.commonpath(label_files)
    # label_files = [os.path.relpath(l, common_path) for l in label_files]
    #
    # for y_i, f_i in zip(y, label_files):
    #     if hasattr(test, 'untransfrom_label'):
    #         y_i = test.untransfrom_label(y_i)
    #     y_i = y_i.transpose(1, 2, 0)
    #     out_file = os.path.join(out, 'test', f_i)
    #     os.makedirs(os.path.dirname(out_file), exist_ok=True)
    #     save_image(out_file, y_i, spacing=None)

def get_patient_index(filename):
    with open(filename) as fp:
        return [l.strip() for l in fp.readlines()]

if __name__ == '__main__':
    main()
