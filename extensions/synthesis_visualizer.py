import copy
import warnings

import cv2
import os
import six
import numpy as np

import chainer
from chainer import configuration
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import function
from chainer import iterators
from chainer import link
from chainer import reporter as reporter_module
from chainer.training import extension

def centerize(src, dst_shape, margin_color=None):
    """Centerize image for specified image size

    @param src: image to centerize
    @param dst_shape: image shape (height, width) or (height, width, channel)
    """
    if src.shape[:2] == dst_shape[:2]:
        return src
    centerized = np.zeros(dst_shape, dtype=src.dtype)
    if margin_color:
        centerized[:, :] = margin_color
    pad_vertical, pad_horizontal = 0, 0
    h, w = src.shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if h < dst_h:
        pad_vertical = (dst_h - h) // 2
    if w < dst_w:
        pad_horizontal = (dst_w - w) // 2
    centerized[pad_vertical:pad_vertical+h,
               pad_horizontal:pad_horizontal+w] = src
    return centerized


def _tile_images(imgs, tile_shape, concatenated_image):
    """Concatenate images whose sizes are same.

    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param concatenated_image: returned image.
        if it is None, new image will be created.
    """
    y_num, x_num = tile_shape
    one_width = imgs[0].shape[1]
    one_height = imgs[0].shape[0]
    if concatenated_image is None:
        if len(imgs[0].shape) == 3:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num, 3), dtype=np.uint8)
        else:
            concatenated_image = np.zeros(
                (one_height * y_num, one_width * x_num), dtype=np.uint8)
    for y in range(y_num):
        for x in range(x_num):
            i = x + y * x_num
            if i >= len(imgs):
                pass
            else:
                concatenated_image[y*one_height:(y+1)*one_height,
                                   x*one_width:(x+1)*one_width, ] = imgs[i]
    return concatenated_image


def get_tile_image(imgs, tile_shape=None, result_img=None, margin_color=None):
    """Concatenate images whose sizes are different.

    @param imgs: image list which should be concatenated
    @param tile_shape: shape for which images should be concatenated
    @param result_img: numpy array to put result image
    """
    from skimage.transform import resize

    def get_tile_shape(img_num):
        x_num = 0
        y_num = int(np.sqrt(img_num))
        while x_num * y_num < img_num:
            x_num += 1
        return x_num, y_num

    if tile_shape is None:
        tile_shape = get_tile_shape(len(imgs))

    # get max tile size to which each image should be resized
    max_height, max_width = np.inf, np.inf
    for img in imgs:
        max_height = min([max_height, img.shape[0]])
        max_width = min([max_width, img.shape[1]])

    # resize and concatenate images
    for i, img in enumerate(imgs):
        h, w = img.shape[:2]
        dtype = img.dtype
        h_scale, w_scale = max_height / h, max_width / w
        scale = min([h_scale, w_scale])
        h, w = int(scale * h), int(scale * w)
        img = resize(img, (h, w), preserve_range=True, mode='constant').astype(dtype)
        if len(img.shape) == 3:
            img = centerize(img, (max_height, max_width, 3), margin_color)
        else:
            img = centerize(img, (max_height, max_width), margin_color)
        imgs[i] = img
    return _tile_images(imgs, tile_shape, result_img)

class CGANValidator(extension.Extension):

    trigger = 1, 'epoch'
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER

    name = None

    def __init__(self, iterator, target, filename,
                 converter=convert.concat_examples, \
                 device=None, eval_hook=None, eval_func=None, \
                 n_valid=None, n_vis=None):

        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator

        if isinstance(target, link.Link):
            target = {'main': target}
        self._targets = target

        self.filename = filename
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func

        self._n_valid = len(iterator['main'].dataset) if n_valid is None else n_valid
        self._n_vis = self._n_valid if n_vis is None else n_vis

        for key, iter in six.iteritems(iterator):
            if (isinstance(iter, (iterators.SerialIterator,
                                  iterators.MultiprocessIterator,
                                  iterators.MultithreadIterator)) and
                    getattr(iter, 'repeat', False)):
                msg = 'The `repeat` property of the iterator {} '
                'is set to `True`. Typically, the validator sweeps '
                'over iterators until they stop, '
                'but as the property being `True`, this iterator '
                'might not stop and evaluation could go into '
                'an infinite loop.'
                'We recommend to check the configuration '
                'of iterators'.format(key)
                warnings.warn(msg)

    def get_iterator(self, name):
        """Returns the iterator of the given name."""
        return self._iterators[name]

    def get_all_iterators(self):
        """Returns a dictionary of all iterators."""
        return dict(self._iterators)

    def get_target(self, name):
        """Returns the target link of the given name."""
        return self._targets[name]

    def get_all_targets(self):
        """Returns a dictionary of all target links."""
        return dict(self._targets)

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
            prefix = ''
        for name, target in six.iteritems(self._targets):
            reporter.add_observer(prefix + name, target)
            reporter.add_observers(prefix + name,
                                   target.namedlinks(skipself=True))

        filename = self.filename.format(trainer)
        filename = os.path.join(trainer.out, filename)
        with reporter:
            with configuration.using_config('train', False):
                result = self.validate(filename)

        reporter_module.report(result)
        return result

    def validate_core(self, *args, **kwargs):

        enc = self._targets['enc']
        dec = self._targets['dec']
        _ = self._targets['dis']

        x, t = args
        z = enc(x)
        y = dec(z)

        acc = chainer.functions.mean_absolute_error(y, t)

        chainer.report({'acc': acc}, enc)
        chainer.report({'acc': acc}, dec)

        # to cpu
        x = x.get() # TODO: redundancy
        t = t.get() # TODO: redundancy
        y = y.data.get()
        e = np.abs(y-t) - 1. # TODO: ad-hoc

        return x, y ,t, e

    def visualize(self, xs, out, pseudo_color=False):
        assert(isinstance(xs, list))

        xs = np.concatenate(xs, axis=0)
        xs = np.clip(xs*128. + 128., 0.0, 255.0) # TODO: ad-hoc
        xs = xs.astype(np.uint8)
        xs = xs.transpose(0, 2, 3, 1)

        if pseudo_color: # only visualize 1st channel
            xs = [cv2.applyColorMap(x[:,:,0], cv2.COLORMAP_JET) for x in xs]
            xs = np.asarray(xs)

        if xs.shape[-1]==1: xs = np.repeat(xs, 3, axis=-1) # to RGB
        if xs.shape[-1]!=3: raise NotImplementedError()

        xs = get_tile_image(xs)

        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, xs)

    def validate(self, filename):
        """Validates the model and returns a result dictionary.
        This method runs the evaluation loop over the validation dataset. It
        accumulates the reported values to :class:`~chainer.DictSummary` and
        returns a dictionary whose values are means computed by the summary.
        Note that this function assumes that the main iterator raises
        ``StopIteration`` or code in the evaluation loop raises an exception.
        So, if this assumption is not held, the function could be caught in
        an infinite loop.
        Users can override this method to customize the evaluation routine.
        .. note::
            This method encloses :attr:`eval_func` calls with
            :func:`function.no_backprop_mode` context, so all calculations
            using :class:`~chainer.FunctionNode`\\s inside
            :attr:`eval_func` do not make computational graphs. It is for
            reducing the memory consumption.
        Returns:
            dict: Result dictionary. This dictionary is further reported via
            :func:`~chainer.report` without specifying any observer.
        """
        iterator = self._iterators['main']
        eval_func = self.eval_func or self.validate_core

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        n_valid = 0
        xs = []
        ys = []
        ts = []
        es = []

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        x, y ,t, e = eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        x, y ,t, e = eval_func(**in_arrays)
                    else:
                        x, y ,t, e = eval_func(in_arrays)

            if len(xs) <= self._n_vis:
                xs.append(x)
                ys.append(y)
                ts.append(t)
                es.append(e)

            n_valid += len(batch)
            if n_valid > self._n_valid: break

            summary.add(observation)

        # visualize
        out, ext = os.path.splitext(filename)
        self.visualize(xs, os.path.join(out + '_x' + ext), False)
        self.visualize(ts, os.path.join(out + '_t' + ext), False)
        self.visualize(ys, os.path.join(out + '_y' + ext), False)
        self.visualize(es, os.path.join(out + '_e' + ext), True)

        return summary.compute_mean()

    def finalize(self):
        """Finalizes the validator object.
        This method calls the `finalize` method of each iterator that
        this validator has.
        It is called at the end of training loops.
        """
        for iterator in six.itervalues(self._iterators):
            iterator.finalize()
