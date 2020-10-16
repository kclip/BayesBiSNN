from optimizer.BiOptimizer import BiOptimizer
import torch
from torch.optim.optimizer import required
from utils.binarize import binarize

class BayesBiSNNRP(BiOptimizer):
    def __init__(self, concrete_binary_params, latent_params, lr=required, temperature=required, device=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, temperature=temperature)
        super(BayesBiSNNRP, self).__init__(concrete_binary_params, latent_params, defaults)
        self.device = device

    @torch.no_grad()
    def step(self):
        for i, group in enumerate(self.binary_param_groups):
            for j, w in enumerate(group['params']):
                if w.grad is None:
                    continue

                mu = torch.tanh(self.param_groups[i]['params'][j].data)
                scale = (1 - w * w + 1e-10) / group['temperature'] / (1 - mu * mu + 1e-10)

                d_w = w.grad
                # print(w.grad.shape, torch.max(torch.abs(d_w * scale)), torch.max(torch.abs(self.param_groups[i]['params'][j])))
                self.param_groups[i]['params'][j].add_(d_w * scale, alpha=-group['lr'])


    def update_concrete_weights(self):
        for i, group in enumerate(self.binary_param_groups):
            for j, w in enumerate(group['params']):
                if w.requires_grad:
                    epsilon = torch.rand(w.data.shape).to(self.device)
                    delta = torch.log(epsilon / (1 - epsilon)) / 2

                    w.data = torch.tanh((delta + self.param_groups[i]['params'][j]) / group['temperature'])

                else:
                    binarize(w)


class BayesBiSNNRF(BiOptimizer):
    def __init__(self, concrete_binary_params, latent_params, lr=required, device=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        self.device = device

        defaults = dict(lr=lr)
        super(BayesBiSNNRF, self).__init__(concrete_binary_params, latent_params, defaults)

    @torch.no_grad()
    def step(self, loss):
        for i, group in enumerate(self.binary_param_groups):
            for j, w in enumerate(group['params']):
                if w.grad is None:
                    # print('///')
                    continue

                sign = torch.sign(w).to(self.device)
                grad_log = sign * (1. / 2. / torch.sigmoid(2. * sign * self.param_groups[i]['params'][j]))
                d_w = loss * grad_log

                # print(w.grad.shape, torch.max(torch.abs(d_w)), torch.max(torch.abs(self.param_groups[i]['params'][j])))

                self.param_groups[i]['params'][j].add_(d_w, alpha=-group['lr'])


    def update_concrete_weights(self):
        for i, group in enumerate(self.binary_param_groups):
            for j, w in enumerate(group['params']):
                if w.requires_grad:
                    w.data = 2 * torch.bernoulli(torch.sigmoid(2 * self.param_groups[i]['params'][j])).to(self.device) - 1
                else:
                    binarize(w)



class BayesBiSNNSTGS(BiOptimizer):
    def __init__(self, concrete_binary_params, latent_params, lr=required, temperature=required, device=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, temperature=temperature)
        super(BayesBiSNNSTGS, self).__init__(concrete_binary_params, latent_params, defaults)
        self.device = device

    @torch.no_grad()
    def step(self):
        for i, group in enumerate(self.binary_param_groups):
            for j, w in enumerate(group['params']):
                if w.grad is None:
                    continue

                mu = torch.tanh(self.param_groups[i]['params'][j].data)

                epsilon = torch.rand(w.data.shape).to(self.device)
                delta = torch.log(epsilon / (1 - epsilon)) / 2
                w_st = torch.tanh((delta + self.param_groups[i]['params'][j]) / group['temperature'])
                scale = (1 - w_st * w_st + 1e-10) / group['temperature'] / (1 - mu * mu + 1e-10)

                d_w = w.grad
                # print(w.grad.shape, torch.max(torch.abs(d_w * scale)), torch.max(torch.abs(self.param_groups[i]['params'][j])))
                self.param_groups[i]['params'][j].add_(d_w * scale, alpha=-group['lr'])


    def update_concrete_weights(self):
        for i, group in enumerate(self.binary_param_groups):
            for j, w in enumerate(group['params']):
                if w.requires_grad:
                    tau = 1e-7
                    epsilon = torch.rand(w.data.shape).to(self.device)
                    delta = torch.log(epsilon / (1 - epsilon)) / 2
                    w.data = torch.tanh((delta + self.param_groups[i]['params'][j]) / tau)

                else:
                    binarize(w)
