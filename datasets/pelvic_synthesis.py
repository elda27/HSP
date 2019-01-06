import numpy as np
import glob
import os
import cv2
import chainer
from utils import load_image
import chainer.functions as F
import cupy


def clip(a, a_min, a_max):

    if isinstance(a, np.ndarray):
        a = np.clip(a, a_min, a_max)
    elif isinstance(a, (chainer.Variable, cupy.ndarray)):
        a = F.clip(a, a_min, a_max)
    else:
        raise NotImplementedError(a.__class__)
    return a


class XpSynthesisDataset(chainer.dataset.DatasetMixin):

    def __init__(self,
                 data_dir,
                 patients,
                 image_ext=r'_image.mhd',
                 label_ext=r'_image.mhd',
                 image_dir=r'Xp/',
                 label_dir=r'DRR/',
                 image_normalize=True,
                 label_normalize=True,
                 image_range=(0., 3500.),
                 image_size=(512, 512),
                 label_range=(0., 255.),
                 label_dtype=np.float32,
                 augmentation=False):

        assert(isinstance(patients, list))

        self.data_dir = data_dir
        self.patients = patients
        self.image_ext = image_ext
        self.label_ext = label_ext
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_normalize = image_normalize
        self.label_normalize = label_normalize
        self.image_range = image_range
        self.image_size = image_size
        self.label_range = label_range
        self.label_dtype = label_dtype
        self.augmentation = augmentation

        assert(isinstance(self.image_size, (type(None), tuple)))

        self.image_files = []
        self.label_files = []

        files = [self.image_files, self.label_files]
        exts = [self.image_ext, self.label_ext]
        tdirs = [self.image_dir, self.label_dir]

        patterns = []
        for p in patients:
            for f, d, e in zip(files, tdirs, exts):
                glob_pattern = os.path.join(data_dir, d, p + '*' + e)
                patterns.append(glob_pattern)
                f.extend(glob.glob(glob_pattern))

        assert len(self.image_files) != 0, patterns[0]
        assert len(self.image_files) == len(self.label_files), patterns[1]

    def __len__(self):
        return len(self.image_files)

    def transform_image(self, image, no_clip=False):
        if self.image_normalize:
            # [0, 1]
            image = (image - self.image_range[0]) / \
                (self.image_range[1]-self.image_range[0])
            image = image * 2.0 - 1.0  # [-1, 1]
            if not no_clip:
                image = clip(image, -1.0, 1.0)
        return image

    def untransfrom_image(self, image, no_clip=False):
        if self.image_normalize:
            image = (image + 1.0) / 2.0  # [0, 1]
            image = image * \
                (self.image_range[1]-self.image_range[0]) + self.image_range[0]
            if not no_clip:
                image = clip(image, self.image_range[0], self.image_range[1])
        return image

    def transform_label(self, label, no_clip=False):
        if self.label_normalize:
            # [0, 1]
            label = (label - self.label_range[0]) / \
                (self.label_range[1]-self.label_range[0])
            label = label * 2.0 - 1.0  # [-1, 1]
            if not no_clip:
                label = clip(label, -1.0, 1.0)
        return label

    def untransfrom_label(self, label, no_clip=False):
        if self.label_normalize:
            label = (label + 1.0) / 2.0  # [0, 1]
            label = label * \
                (self.label_range[1]-self.label_range[0]) + self.label_range[0]
            if not no_clip:
                label = clip(label, self.label_range[0], self.label_range[1])

        return label

    def data_augment(self, image, label, crop_size, flip=True):
        assert(isinstance(crop_size, (list, tuple)))
        _, h, w = image.shape
        x_s = np.random.randint(0, h-crop_size[0])
        x_e = x_s + crop_size[0]
        y_s = np.random.randint(0, w-crop_size[1])
        y_e = y_s + crop_size[1]

        image = image[:, y_s:y_e, x_s:x_e]
        label = label[:, y_s:y_e, x_s:x_e]

        if flip and np.random.rand() > 0.5:
            image = image[:, :, ::-1]
            label = label[:, :, ::-1]

        return image, label

    def get_example(self, i):

        # load images
        image_file = self.image_files[i]
        image, _ = load_image(image_file)

        label_file = self.label_files[i]
        label, _ = load_image(label_file)

        # set image size
        image_size = image.shape[1:] if self.image_size is None else self.image_size
        if self.augmentation:
            crop_size = image_size
            image_size = tuple([int(s*1.1) for s in image_size])

        # resize
        image = cv2.resize(image, image_size)
        image = image.astype(np.float32)

        label = cv2.resize(label, image_size)
        label = label.astype(self.label_dtype)

        # transform
        image = self.transform_image(image)
        image = image.astype(np.float32)
        image = image[np.newaxis, :]

        label = self.transform_label(label)
        label = label.astype(self.label_dtype)
        label = label[np.newaxis, :]

        if image.ndim != 3:
            raise ValueError('image.ndim != 3')
        if label.ndim != 3:
            raise ValueError('label.ndim != 3')

        # augmentation
        if self.augmentation:
            image, label = self.data_augment(image, label, crop_size)

        return image, label


if __name__ == '__main__':

    dataset = XpSynthesisDataset(r'data/xp_synthesis', ['K4287'])
    image, label = dataset.get_example(0)

    print(dataset.image_files[0])
    print(np.min(image), np.max(image))
    print(np.min(label), np.max(label))

    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.imshow(image[0])
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(label[0])
    plt.colorbar()
    plt.show()

    image = dataset.untransfrom_image(image)
    label = dataset.untransfrom_label(label)
    print(np.min(image), np.max(image))
    print(np.min(label), np.max(label))
