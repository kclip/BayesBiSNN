from model.LIF_base import *
from utils.activations import smooth_step
from torch.nn.init import _calculate_correct_fan, calculate_gain
from utils.misc import get_scale

class LIFMLP(LIFNetwork):
    def __init__(self,
                 input_shape,
                 output_shape,
                 n_neurons=[128],
                 tau_mem=[10],
                 tau_syn=[6],
                 tau_ref=[2],
                 prior_p=0.5,
                 activation=smooth_step,
                 num_layers=1,
                 lif_layer_type=LIFLayer,
                 with_bias=True,
                 scaling=True,
                 softmax=True
                 ):

        self.softmax = softmax
        self.scales = []  # Scaling factors for readout layers

        num_layers = max(num_layers, max([len(n_neurons), len(tau_mem), len(tau_syn), len(tau_ref)]))
        self.num_layers = num_layers

        if len(n_neurons) == 1:
            n_neurons = n_neurons * num_layers
        if len(tau_mem) == 1:
            tau_mem = tau_mem * num_layers
        if len(tau_syn) == 1:
            tau_syn = tau_syn * num_layers
        if len(tau_ref) == 1:
            tau_ref = tau_ref * num_layers

        super(LIFMLP, self).__init__()

        if len(input_shape) > 1:
            input_shape = [np.prod(input_shape)]

        Mhid = input_shape + n_neurons

        # Creates LIF linear layers
        for i in range(num_layers):
            base_layer = nn.Linear(Mhid[i], Mhid[i+1], bias=with_bias)
            # Initialize weights in {-10, 10} with probas following a Bernoulli distribution and probability prior_
            base_layer.weight.data[:] = (2 * torch.bernoulli(torch.ones(base_layer.weight.shape) * prior_p) - 1) * 10
            if with_bias:
                base_layer.bias.data[:] = (2 * torch.bernoulli(torch.ones(base_layer.bias.shape) * prior_p) - 1) * 10

            layer = lif_layer_type(base_layer,
                                   activation=activation,
                                   tau_mem=tau_mem[i],
                                   tau_syn=tau_syn[i],
                                   tau_ref=tau_ref[i],
                                   scaling=scaling,
                                   )

            readout = nn.Linear(Mhid[i+1], output_shape, bias=with_bias)
            # Initialize readout weights in {-1, 1} with probas following a Bernoulli distribution and probability prior_p
            readout.weight.data[:] = (2 * torch.bernoulli(torch.ones(readout.weight.shape) * prior_p) - 1) * 10
            if with_bias:
                readout.bias.data[:] = (2 * torch.bernoulli(torch.ones(readout.bias.shape) * prior_p) - 1) * 10
            for param in readout.parameters():
                param.requires_grad = False

            self.LIF_layers.append(layer)
            self.readout_layers.append(readout)
            self.scales.append(get_scale(readout, scaling))


    def forward(self, inputs):
        s_out = []
        r_out = []
        u_out = []

        # Forward propagates the signal through all layers
        inputs = inputs.view(inputs.size(0), -1)
        for lif, ro, scale in zip(self.LIF_layers, self.readout_layers, self.scales):
            s, u = lif(inputs)
            r_ = ro(s) * scale  # Readout outputs are scaled down
            inputs = s.detach()  # Gradients are not propagated through the layers
            s_out.append(s)

            if self.softmax:
                r_ = torch.softmax(r_, dim=-1)
            r_out.append(r_)
            u_out.append(u)

        return s_out, r_out, u_out
