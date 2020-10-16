from optimizer.BiOptimizer import BiOptimizer
import torch
from torch.optim.optimizer import required
from utils.binarize import binarize, binarize_stochastic, clip


class BiSGD(BiOptimizer):
    def __init__(self, binary_params, latent_params, lr=required, binarizer=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, binarizer=binarizer)
        super(BiSGD, self).__init__(binary_params, latent_params, defaults)

    @torch.no_grad()
    def step(self):
        for i, group in enumerate(self.binary_param_groups):
            for j, p in enumerate(group['params']):
                if p.grad is None:
                    group['binarizer'](p)
                    continue

                d_p = p.grad
                clip(d_p)
                # print(p.grad.shape, torch.max(torch.abs(d_p)), torch.max(torch.abs(self.param_groups[i]['params'][j])))

                self.param_groups[i]['params'][j].add_(d_p, alpha=-group['lr'])

                p.data.copy_(self.param_groups[i]['params'][j])
                group['binarizer'](p)
