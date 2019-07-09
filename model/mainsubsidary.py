import math
import os

import torch.nn as nn
import torch
import torch.nn.functional as F

from torch.utils.cpp_extension import load

cur_dir_path = os.path.dirname(__file__)

binactive_cuda = load(
    'binactive_cuda', [os.path.join(cur_dir_path, 'cuda/binactive_cuda.cpp'), os.path.join(cur_dir_path, 'cuda/binactive_cuda_kernel.cu')], extra_cflags=['-O3'])

class binactive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        input = binactive_cuda.forward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = binactive_cuda.backward(grad_input, input)
        return grad_input

class BinActive(nn.Module):
    def __init__(self):
        super(BinActive, self).__init__()

    def forward(self, x):
        return binactive.apply(x)

class EqualActive(nn.Module):
    def __init__(self):
        super(EqualActive,self).__init__()
    def forward(self,x):
        output = x
        return output
    
    
class MainSubConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
                 kernel_size=-1, stride=-1, padding=0, dropout=0, bias = None, main_or_sub=False):
        super(MainSubConv2d, self).__init__()
        self.layer_type = 'MainSubConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.weight = nn.Parameter(torch.Tensor(output_channels, input_channels, kernel_size, kernel_size),
                                   requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_channels))
        else:
            self.register_parameter('bias', None)
        if dropout != 0:
            self.dropout = nn.Dropout(dropout)
        
        self.mask = torch.nn.Parameter(torch.ones(self.weight.size(0),1,1,1),requires_grad=True)
        self.main_or_sub = main_or_sub
        self.reset_parameters()
      
        
    def reset_parameters(self):
        n = self.input_channels
        n *= (self.kernel_size ^ 2)
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.mask.data.uniform_(-10e-5,10e-5)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)



    def forward(self, x):
        x = BinActive()(x)
        if self.dropout_ratio != 0:
            x = self.dropout(x) 
        if not self.main_or_sub:
            self.filtermask = BinActive()(self.mask)
            self.filtermask = (self.filtermask - 1)/2
        else:
            self.filtermask = EqualActive()(self.mask)
        self._weight = self.weight*self.filtermask + self.weight
        return F.conv2d(x, self._weight, self.bias, self.stride, self.padding)

    def extra_repr(self):
        s = ('{input_channels}, {output_channels}, kernel_size={kernel_size}'
             ', stride={stride}, binact={binact}')
        return s.format(**self.__dict__)
