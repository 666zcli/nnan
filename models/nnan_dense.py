import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['nnan_denseunit']


class NNaNUnit(nn.Module):
    """
    Args:
        dims: the list of numbers of neurons
    """
    def __init__(self, dims):
        super(NNaNUnit, self).__init__()
        assert(len(dims)>0)
        pad_dims = [1] + dims + [1]
        for idx, dim in enumerate(pad_dims[:-1]):
            self.add_module('Linear'+str(idx), nn.Linear(dim, pad_dims[idx + 1]))
            if idx  < len(dims):
                self.add_module('ReLU'+str(idx), nn.ReLU(True))

    def forward(self, inputs):
        # reshape to a vector and compute
        orig_shape = inputs.size()
       
        outputs = inputs.view(torch.numel(inputs), 1)
        #output = outputs
        for module in self._modules.values():

            outputs = module(outputs)
                        
        # reshape back to the original shape
        return outputs.view(orig_shape) + inputs


def nnan_denseunit(**kwargs):
    return NNaNUnit(**kwargs)
