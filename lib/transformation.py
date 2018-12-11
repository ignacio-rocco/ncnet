from __future__ import print_function, division
import os
import sys
from skimage import io
import pandas as pd
import numpy as np
import torch
from torch.nn.modules.module import Module
from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn.functional as F

from lib.torch_util import expand_dim

class AffineTnf(object):
    def __init__(self, out_h=240, out_w=240, use_cuda=True):
        self.out_h = out_h
        self.out_w = out_w
        self.use_cuda = use_cuda
        self.gridGen = AffineGridGen(out_h=out_h, out_w=out_w, use_cuda=use_cuda)
        self.theta_identity = torch.Tensor(np.expand_dims(np.array([[1,0,0],[0,1,0]]),0).astype(np.float32))
        if use_cuda:
            self.theta_identity = self.theta_identity.cuda()

    def __call__(self, image_batch, theta_batch=None, out_h=None, out_w=None):
        if image_batch is None:
            b=1
        else:
            b=image_batch.size(0)
        if theta_batch is None:
            theta_batch = self.theta_identity
            theta_batch = theta_batch.expand(b,2,3).contiguous()
            theta_batch = Variable(theta_batch,requires_grad=False)        
        
        # check if output dimensions have been specified at call time and have changed
        if (out_h is not None and out_w is not None) and (out_h!=self.out_h or out_w!=self.out_w):
            gridGen = AffineGridGen(out_h, out_w)
        else:
            gridGen = self.gridGen
        
        sampling_grid = gridGen(theta_batch)
        
        # sample transformed image
        warped_image_batch = F.grid_sample(image_batch, sampling_grid)
        
        return warped_image_batch
    

class AffineGridGen(Module):
    def __init__(self, out_h=240, out_w=240, out_ch = 3, use_cuda=True):
        super(AffineGridGen, self).__init__()        
        self.out_h = out_h
        self.out_w = out_w
        self.out_ch = out_ch
        
    def forward(self, theta):
        b=theta.size()[0]
        if not theta.size()==(b,2,3):
            theta = theta.view(-1,2,3)
        theta = theta.contiguous()
        batch_size = theta.size()[0]
        out_size = torch.Size((batch_size,self.out_ch,self.out_h,self.out_w))
        return F.affine_grid(theta, out_size)
