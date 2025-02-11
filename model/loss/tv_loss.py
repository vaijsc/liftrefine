import torch
import torch.nn as nn
from copy import deepcopy


def tv_loss(tensor, dims, power=1):
    shape = list(tensor.size())
    diffs = []
    for dim in dims:
        pad_shape = deepcopy(shape)
        pad_shape[dim] = 1
        diffs.append(torch.cat([torch.diff(tensor, dim=dim), tensor.new_zeros(pad_shape)], dim=dim))
    return torch.stack(diffs, dim=0).norm(dim=0).pow(power).mean(dim=dims)


class TVLoss(nn.Module):

    def __init__(self,
                 dims=[-2, -1],
                 power=1):
        super().__init__()
        self.dims = dims
        self.power = power

    def forward(self, tensor, weight=None):
        return tv_loss(
            tensor, self.dims, power=self.power) 