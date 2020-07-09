from optimizer.BiOptimizer import BiOptimizer
import torch
from torch.optim.optimizer import required


class BayesBiSNN(BiOptimizer):
    def __init__(self, concrete_binary_params, latent_params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(BayesBiSNN, self).__init__(concrete_binary_params, latent_params, defaults)

    @torch.no_grad()
    def step(self):
        for i, group in enumerate(self.binary_param_groups):
            for j, w in enumerate(group['params']):
                mu = torch.tanh(self.param_groups[i]['params'][j].data)
                scale = (1 - w * w + 1e-10) / self.defaults['temperature'] / (1 - mu * mu + 1e-10)

                if w.grad is None:
                    continue
                d_w = w.grad

                self.param_groups[i]['params'][j].add_(d_w * scale, alpha=-group['lr'])
        self.generate_concrete_weights()


    def generate_concrete_weights(self):
        for i, group in enumerate(self.binary_param_groups):
            for j, w in enumerate(group['params']):
                epsilon = torch.rand_like(w.data)
                delta = torch.log(epsilon / (1 - epsilon)) / 2

                w.data = torch.tanh((delta + self.param_groups[i]['params'][j]) / group['temperature'])
