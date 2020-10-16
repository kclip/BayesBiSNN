import torch

def binarize(x, magnitude=1.):
    x.data[x.data >= 0] = magnitude
    x.data[x.data < 0] = - magnitude

def hard_sigmoid(x):
    return torch.max(torch.zeros(x.shape), torch.min(torch.ones(x.shape), (x + 1)/2))

def binarize_stochastic(x):
    x.data = hard_sigmoid(x)

def clip(x):
    x = torch.max(torch.ones(x.shape).to(x.device) * (-1), torch.min(torch.ones(x.shape).to(x.device), x))



