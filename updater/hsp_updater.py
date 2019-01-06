import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from PIL import Image

from chainer.backends import cuda
from chainer import function
from chainer.utils import type_check
import numpy


class HspUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
