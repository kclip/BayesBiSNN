import torch
import numpy as np

class SmoothStep(torch.autograd.Function):
    '''
    Modified from: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    '''

    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return (x >=0).float()

    def backward(aux, grad_output):
        input, = aux.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= -.5] = 0
        grad_input[input > .5] = 0
        return grad_input


class SigmoidStep(torch.autograd.Function):
    @staticmethod
    def forward(aux, x):
        aux.save_for_backward(x)
        return (x >= 0).type(x.dtype)

    def backward(aux, grad_output):
        input, = aux.saved_tensors
        res = torch.sigmoid(input)
        return res*(1-res)*grad_output


smooth_step = SmoothStep().apply
smooth_sigmoid = SigmoidStep().apply
