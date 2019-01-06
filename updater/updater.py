#!/usr/bin/env python

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from PIL import Image

from chainer import cuda
from chainer import function
from chainer.utils import type_check
import numpy

class CGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        super(CGANUpdater, self).__init__(*args, **kwargs)

    def loss_enc(self, enc, x_out, t_out, y_out, lam1=100, lam2=1):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, enc)
        return loss

    def loss_dec(self, dec, x_out, t_out, y_out, lam1=100, lam2=1):
        batchsize,_,w,h = y_out.data.shape
        loss_rec = lam1*(F.mean_absolute_error(x_out, t_out))
        loss_adv = lam2*F.sum(F.softplus(-y_out)) / batchsize / w / h
        loss = loss_rec + loss_adv
        chainer.report({'loss': loss}, dec)
        return loss


    def loss_dis(self, dis, y_in, y_out):
        batchsize,_,w,h = y_in.data.shape

        L1 = F.sum(F.softplus(-y_in)) / batchsize / w / h
        L2 = F.sum(F.softplus(y_out)) / batchsize / w / h
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss

    def update_core(self):
        enc_optimizer = self.get_optimizer('enc')
        dec_optimizer = self.get_optimizer('dec')
        dis_optimizer = self.get_optimizer('dis')

        enc = enc_optimizer.target
        dec = dec_optimizer.target
        dis = dis_optimizer.target

        batch = self.get_iterator('main').next()
        in_arrays = self.converter(batch, self.device)
        x, t = in_arrays

        z = enc(x)
        y = dec(z)

        dis_fake = dis(x, y)
        dis_real = dis(x, t)

        enc_optimizer.update(self.loss_enc, enc, y, t, dis_fake)
        for z_i in z: # z has hierarchical hidden variables
            z_i.unchain_backward()

        dec_optimizer.update(self.loss_dec, dec, y, t, dis_fake)
        y.unchain_backward()

        dis_optimizer.update(self.loss_dis, dis, dis_real, dis_fake)
