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


def one_hot_crossentropy(input, label):
    label = torch.argmax(label, dim=-1)
    return torch.nn.CrossEntropyLoss()(input, label)

class DECOLLELoss(object):
    def __init__(self, loss_fn, net, reg_l=None):
        self.nlayers = len(net)
        if len(loss_fn) == 1:
            loss_fn *= self.nlayers
        self.loss_fn = loss_fn
        self.num_losses = len([l for l in loss_fn if l is not None])
        assert len(loss_fn) == self.nlayers, "Mismatch is in number of loss functions and layers. You need to specify one loss functino per layer"
        self.reg_l = reg_l
        if self.reg_l is None:
            self.reg_l = [0 for _ in range(self.nlayers)]

    def __len__(self):
        return self.nlayers

    def __call__(self, s, r, u, target, mask=1, sum_=True):
        loss_tv = []
        for i, loss_layer in enumerate(self.loss_fn):
            if loss_layer is not None:
                loss_tv.append(loss_layer(r[i] * mask, target * mask))
                if self.reg_l[i] > 0:
                    uflat = u[i].reshape(u[i].shape[0], -1)
                    reg1_loss = self.reg_l[i] * 1e-2 * (torch.nn.functional.relu(uflat + .01)).mean()
                    reg2_loss = self.reg_l[i] * 6e-5 * torch.nn.functional.relu((.1-torch.sigmoid(uflat)).mean())
                    loss_tv[-1] += reg1_loss + reg2_loss
        # print(loss_tv)
        if sum_:
            return sum(loss_tv)
        else:
            return loss_tv
