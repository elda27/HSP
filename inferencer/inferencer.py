import chainer
import chainer.functions as F
from chainer.backends import cuda
import numpy as np

class Inferencer:
    def __init__(self, model, iterator, device=-1):
        self.model = model
        self.iterator = iterator
        self.device = device

    def __call__(self):
        for x in self.iterator:
            x.to_gpu()

            y = F.argmax(self.model(x), axis=1)

            xp = cuda.get_array_module(x)
            y = xp.asnumpy(y)
            for i in range(y.shape[0]):
                yield y[i]
