import argparse
import os
import glob
import numpy as np
import re
from pathlib import Path
from itertools import zip_longest

import chainer
import models
import mhd

from datasets import Dataset
from utils import get_training_log_dir
from inferencer import Inferencer

def main():
    parser = argparse.ArgumentParser(description='test for hsp')
    parser.add_argument('-b', '--batchsize', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('-g', '--gpu', type=int, default=1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('-o', '--output-dir', default='test',
                        help='Directory to output the result')

    parser.add_argument('--test-index', type=str, default=None,
                        help='Index file of the test inference')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input directory of the dataset or pattern of the input images.')
    parser.add_argument('--spacing', nargs=3, type=float, default=(1.0, 1.0, 1.0))

    parser.add_argument('--log-dir', default='./log', help='Root directory of the training log.')
    parser.add_argument('--log-index', default=None, type=int,
                        help='Prefix index of the log directory')
    parser.add_argument('--model', type=str, default=None,
                        help='Saved model path.')

    parser.add_argument('-o', '--output-dir', type=str, default='./out')

    args = parser.parse_args()
    args.image_size = tuple(args.image_size)

    # setup models
    predictor = models.HierarchicalSurfacePredictor()

    # load pre-trained model
    chainer.serializers.load_npz(
        get_model_path(**vars(args)),
        predictor
    )

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        predictor.to_gpu()

    # setup dataset
    file_pattern, dataset = make_dataset(**vars(args))
    print('number of images:', len(dataset))

    iterator = chainer.iterators.MultiprocessIterator(
        dataset, args.batchsize,
        repeat=False, shuffle=False
    )

    # setup a inferencer
    infer = Inferencer(predictor, iterator, device=args.gpu)
    infer_iter = zip_longest(
        dataset.files.get('labels', []),
        dataset.files.get('projections', []),
        infer()
    )

    input = Path(args.input)
    file_pattern = construct_regex_from_file_pattern(file_pattern)

    output_dir = Path(args.out)
    os.makedirs(str(output_dir), exist_ok=True)

    for t_file, x_file, ys in infer_iter:
        matched = re.match(file_pattern, x_file)
        case_vars = get_matched_array(matched)
        output_path = output_dir / args.name.format(
            *case_vars
        )
        case_name = '_'.join(case_vars)
        cur_output_dir =  output_dir / case_name
        cur_output_dir.mkdir(exist_ok=True)

        mhd_copy(x_file, cur_output_dir / 'projections.mhd')

        if t_file is None:
            spacing = args.spacing
        else:
            header = mhd_copy(str(t_file), cur_output_dir / 'input_label.mhd')
            spacing = header['ElementSpacing']

        mhd.write(cur_output_dir / 'reconstruct_label', ys, {'ElementSpacing':spacing})

def mhd_copy(source_file, dest_file):
    source, header = mhd.read(str(source_file))
    del header['ElementDataFile']
    mhd.write(dest_file, source, header)
    return header


def get_matched_array(matched):
    index = 0
    values = []
    try:
        while True:
            index += 1
            values.append(
                matched.group(
                    'P{:03d}'.format(index)
                )
            )
    except IndexError:
        pass
    return values


def construct_regex_from_file_pattern(file_pattern):
    file_pattern = file_pattern.replace('.', r'\.')

    target = file_pattern
    pos = -1
    index = 0
    while index != 0 and pos != -1:
        index += 1
        file_pattern.replace('*', '(?P<P{:03d}>.*)'.format(index), 1)
        pos = target[pos + 1:].find('*')

    return re.compile(file_pattern)

def make_dataset(**kwargs):
    if 'test_index' in kwargs:
        patients = get_patient_index(kwargs.get('test_index'))
    else:
        patients = None

    input = Path(kwargs.get('input'))

    if input.is_dir():
        file_pattern = '*_projections.mhd'
        dataset = Dataset(
            str(input),
            patients,
            with_spacing=True
        )
    else:
        file_pattern = input.name
        dataset = Dataset(
            input.parent,
            patients,
            patterns=input.name,
            with_spacing=True
        )
    return file_pattern, dataset


def get_model_path(**kwargs):
    model_path = kwargs.get('model', None)
    if model_path is not None:
        return model_path

    return get_training_log_dir(
        kwargs.get('log_dir'),
        kwargs.get('log_index'),
        test=True
    )


def get_patient_index(filename):
    with open(filename) as fp:
        return [l.strip() for l in fp.readlines()]


if __name__ == '__main__':
    main()
