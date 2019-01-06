#!/usr/bin/env python
from __future__ import print_function
import argparse
import os
import glob
import numpy as np

import chainer
from models.synthesis import Encoder, Decoder, Generator
from inference import Inferencer
from utils import load_image, save_image
import cv2


def transform_image(image, image_range, no_clip=False):
    image = (image - image_range[0]) / \
        (image_range[1]-image_range[0])  # [0, 1]
    image = image * 2.0 - 1.0  # [-1, 1]
    if not no_clip:
        image = np.clip(image, -1.0, 1.0)
    return image


def untransfrom_label(label, label_range, no_clip=False):
    label = (label + 1.0) / 2.0  # [0, 1]
    label = label * (label_range[1]-label_range[0]) + label_range[0]
    if not no_clip:
        label = np.clip(label, label_range[0], label_range[1])
    return label


class Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, image_files, transform_image):
        self.image_files = image_files
        self.transform_image = transform_image
        self.spacing = [None]*len(image_files)

    def __len__(self):
        return len(self.image_files)

    def get_example(self, i):
        f = self.image_files[i]
        image, spacing = load_image(f)
        self.spacing[i] = spacing
        if image.ndim == 3:
            image = np.mean(image, axis=-1)
        image = cv2.resize(image, (512, 512))
        image = image[np.newaxis]
        # image = self.transform_image(image, (0., 3500.))
        return image.astype(np.float32)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='test for synthesis')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--images', '-i', default='data/xp_synthesis/*/Xp_image.mhd',  # for example
                        help='Directory of image files.')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--model_dir', default='',
                        help='Directory to pre-trained model')
    parser.add_argument('--image_size', required=True, nargs=2,
                        type=int, help='')
    args = parser.parse_args()
    args.image_size = tuple(args.image_size)

    # setup models
    enc = Encoder(in_ch=1)
    dec = Decoder(out_ch=1)

    # load pre-trained model
    enc_file = glob.glob(os.path.join(args.model_dir, 'enc_iter_*.npz'))[-1]
    dec_file = glob.glob(os.path.join(args.model_dir, 'dec_iter_*.npz'))[-1]

    chainer.serializers.load_npz(enc_file, enc)
    chainer.serializers.load_npz(dec_file, dec)

    model = Generator(enc, dec)

    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # setup dataset
    image_files = glob.glob(args.images)
    print('number of images:', len(image_files))

    dataset = Dataset(image_files, transform_image)

    iterator = chainer.iterators.SerialIterator(dataset, args.batchsize,
                                                repeat=False, shuffle=False)

    # setup a inferencer
    infer = Inferencer(iterator, model, device=args.gpu)
    y = infer.run()

    # save images
    common_path = os.path.commonpath(image_files)
    out_files = [os.path.relpath(f, common_path) for f in image_files]
    spacing = dataset.spacing

    for y_i, f_i, s_i in zip(y, out_files, spacing):

        #y_i = untransfrom_label(y_i, (0.,255.))
        y_i = (y_i + 1) / 2
        y_i = y_i.transpose(1, 2, 0)
        y_i = y_i[:, :, 0]  # .astype(np.uint8)

        y_i = cv2.resize(y_i, args.image_size)

        out = os.path.join(args.out, f_i)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        save_image(out, y_i, s_i)
