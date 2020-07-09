import torch
import numpy as np
from torch.nn.modules.loss import _Loss


def vrdistance(f, g, tau):
    assert f.shape == g.shape, 'Spike trains must have the same shape, had f: ' + str(f.shape) + ', g: ' + str(g.shape)

    kernel = torch.FloatTensor(np.exp([-t / tau for t in range(f.shape[-1])])).flip(0)  # Flip because conv1d computes cross-correlations
    diff = (f - g)
    return torch.norm(torch.matmul(diff, kernel), dim=0)


class VRDistance(_Loss):
    def __init__(self, tau, size_average=None, reduce=None, reduction='mean'):
        super(VRDistance, self).__init__(size_average, reduce, reduction)
        self.tau = tau

    def forward(self, input, target):
        return vrdistance(input, target, self.tau)


def decolle_loss(r, tgt, loss):
    loss_tv = 0
    for i in range(len(r)):
        loss_tv += loss(r[i], tgt)
    return loss_tv
