import torch.nn as nn
from torch.nn import functional as F
import torch
import numpy as np
from collections import namedtuple
from itertools import chain
import warnings
from utils.misc import get_output_shape
import math
from torch.nn.init import _calculate_correct_fan, calculate_gain

""""
LIF SNN with local errors
Adapted from https://github.com/nmi-lab/decolle-public
"""

dtype = torch.float32


class LIFLayer(nn.Module):
    NeuronState = namedtuple('NeuronState', ['P', 'Q', 'R', 'S'])

    def __init__(self, layer, activation, tau_mem=10, tau_syn=2, tau_ref=8, scaling=True):
        super(LIFLayer, self).__init__()
        self.base_layer = layer

        self.alpha = torch.exp(torch.FloatTensor([-1/tau_mem])).to(self.base_layer.weight.device)
        self.beta = torch.exp(torch.FloatTensor([-1/tau_syn])).to(self.base_layer.weight.device)
        self.alpharp = torch.exp(torch.FloatTensor([-1/tau_ref])).to(self.base_layer.weight.device)

        self.state = None
        self.activation = activation

        if scaling:
            fan = _calculate_correct_fan(layer.weight, mode='fan_in')
            gain = calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
            std = gain / math.sqrt(fan)
            self.scale = (math.sqrt(3.0) * std)
        #     if type(layer) == nn.Conv2d:
        #         n = layer.in_channels
        #         for k in layer.kernel_size:
        #             n *= k
        #         self.scale = np.sqrt(3. * layer.groups / n)
        #
        #     elif hasattr(layer, 'in_features'):
        #         self.scale = 1. / np.prod(self.base_layer.in_features)
        else:
            self.scale = 1.


    def cuda(self, device=None):
        '''
        Handle the transfer of the neuron state to cuda
        '''
        self = super().cuda(device)
        self.state = None
        self.base_layer = self.base_layer.cuda()
        return self

    def cpu(self, device=None):
        '''
        Handle the transfer of the neuron state to cpu
        '''
        self = super().cpu(device)
        self.state = None
        self.base_layer = self.base_layer.cpu()
        return self

    @staticmethod
    def reset_parameters(layer):
        if type(layer) == nn.Conv2d:
            conv_layer = layer
            n = conv_layer.in_channels
            for k in conv_layer.kernel_size:
                n *= k
            stdv = 1. / np.sqrt(n) / 250
            conv_layer.weight.data.uniform_(-stdv * 1e-2, stdv * 1e-2)
            if conv_layer.bias is not None:
                conv_layer.bias.data.uniform_(-stdv, stdv)
        # elif hasattr(layer, 'out_features'):
        #     layer.weight.data[:] *= 0
        #     if layer.bias is not None:
        #         layer.bias.data[:] *= 0
        # layer.bias.data.uniform_(-1e-3, 1e-3)
        # else:
        #     warnings.warn('Unhandled data type, not resetting parameters')


    @staticmethod
    def get_out_channels(layer):
        '''
        Wrapper for returning number of output channels in a LIFLayer
        '''
        if hasattr(layer, 'out_features'):
            return layer.out_features
        elif hasattr(layer, 'out_channels'):
            return layer.out_channels
        elif hasattr(layer, 'get_out_channels'):
            return layer.get_out_channels()
        else:
            raise Exception('Unhandled base layer type')


    @staticmethod
    def get_out_shape(layer, input_shape):
        if hasattr(layer, 'out_channels'):
            return get_output_shape(input_shape,
                                    kernel_size=layer.kernel_size,
                                    stride=layer.stride,
                                    padding=layer.padding,
                                    dilation=layer.dilation)
        elif hasattr(layer, 'out_features'):
            return []
        elif hasattr(layer, 'get_out_shape'):
            return layer.get_out_shape()
        else:
            raise Exception('Unhandled base layer type')


    def init_state(self, input_shape):
        device = self.base_layer.weight.device
        out_ch = self.get_out_channels(self.base_layer)
        out_shape = self.get_out_shape(self.base_layer, input_shape)
        self.state = self.NeuronState(P=torch.zeros(input_shape).type(dtype).to(device),
                                      Q=torch.zeros(input_shape).type(dtype).to(device),
                                      R=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device),
                                      S=torch.zeros([input_shape[0], out_ch] + out_shape).type(dtype).to(device))

        self.alpha = self.alpha.to(self.base_layer.weight.device)
        self.beta = self.beta.to(self.base_layer.weight.device)
        self.alpharp = self.alpharp.to(self.base_layer.weight.device)


    def init_parameters(self):
        self.reset_parameters(self.base_layer)


    def forward(self, Sin_t):
        if self.state is None:
            self.init_state(list(Sin_t.shape))

        P = self.alpha * self.state.P + self.state.Q
        Q = self.beta * self.state.Q + Sin_t
        R = self.alpharp * self.state.R + self.state.S
        U = self.base_layer(P) * self.scale - R
        S = self.activation(U)

        self.state = self.NeuronState(P=P.detach(), Q=Q.detach(), R=R.detach(), S=S.detach())
        return S, U


    def get_output_shape(self, input_shape):
        layer = self.base_layer
        if isinstance(layer, nn.Conv2d):
            im_height = input_shape[-2]
            im_width = input_shape[-1]
            height = int((im_height + 2 * layer.padding[0] - layer.dilation[0] *
                          (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1)
            weight = int((im_width + 2 * layer.padding[1] - layer.dilation[1] *
                          (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1)
            return [height, weight]
        else:
            return layer.out_features

    def get_device(self):
        return self.base_layer.weight.device


class LIFNetwork(nn.Module):
    requires_init = True

    def __init__(self):
        super(LIFNetwork, self).__init__()
        self.LIF_layers = nn.ModuleList()
        self.readout_layers = nn.ModuleList()



    def __len__(self):
        return len(self.LIF_layers)


    def forward(self, input):
        raise NotImplemented('')


    @property
    def output_layer(self):
        return self.readout_layers[-1]


    def get_trainable_parameters(self, layer=None):
        if layer is None:
            return chain(*[l.parameters() for l in self.LIF_layers])
        else:
            return self.LIF_layers[layer].parameters()


    def init(self, data_batch=None, burnin=0):

        """
        It is necessary to reset the state of the network whenever a new batch is presented
        """

        if self.requires_init is False:
            return
        for l in self.LIF_layers:
            l.state = None

        if data_batch is not None:
            for i in range(max(len(self), burnin)):
                self.forward(data_batch[i])


    def init_parameters(self):
        for i, l in enumerate(self.LIF_layers):
            l.init_parameters()


    def reset_lc_parameters(self, layer, lc_ampl):
        stdv = lc_ampl / np.sqrt(layer.weight.size(1))
        layer.weight.data.uniform_(-stdv, stdv)
        if layer.bias is not None:
            layer.bias.data.uniform_(-stdv, stdv)


    def get_input_layer_device(self):
        if hasattr(self.LIF_layers[0], 'get_device'):
            return self.LIF_layers[0].get_device()
        else:
            return list(self.LIF_layers[0].parameters())[0].device


    def get_output_layer_device(self):
        return self.output_layer.weight.device
