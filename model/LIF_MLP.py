from model.LIF_base import *
from utils.activations import smooth_step, smooth_sigmoid
from torch.nn.init import _calculate_correct_fan, calculate_gain


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
                 lc_ampl=.5,
                 with_output_layer=True,
                 with_bias=True,
                 scaling=True
                 ):

        self.scales = []
        self.with_output_layer = with_output_layer
        if with_output_layer:
            n_neurons += [output_shape]

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

        for i in range(num_layers):
            base_layer = nn.Linear(Mhid[i], Mhid[i+1], bias=with_bias)
            base_layer.weight.data[:] = (2 * torch.bernoulli(torch.ones(base_layer.weight.shape) * prior_p) - 1) * 10  # / Mhid[i]
            if with_bias:
                base_layer.bias.data[:] = (2 * torch.bernoulli(torch.ones(base_layer.bias.shape) * prior_p) - 1) * 10

            layer = lif_layer_type(base_layer,
                                   activation=activation,
                                   tau_mem=tau_mem[i],
                                   tau_syn=tau_syn[i],
                                   tau_ref=tau_ref[i],
                                   scaling=scaling,
                                   )

            if self.with_output_layer and (i+1 == num_layers):
                readout = nn.Identity()
                # layer.activation = torch.sigmoid
            else:
                readout = nn.Linear(Mhid[i+1], output_shape, bias=with_bias)
                readout.weight.data[:] = (2 * torch.bernoulli(torch.ones(readout.weight.shape) * prior_p) - 1) * 10
                if with_bias:
                    readout.bias.data[:] = (2 * torch.bernoulli(torch.ones(readout.bias.shape) * prior_p) - 1) * 10

                for param in readout.parameters():
                    param.requires_grad = False

            self.LIF_layers.append(layer)
            self.readout_layers.append(readout)
            if scaling:
                fan = _calculate_correct_fan(readout.weight, mode='fan_in')
                gain = calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
                std = gain / math.sqrt(fan)
                self.scales.append(math.sqrt(3.0) * std)

                # self.scales.append(1. / np.prod(readout.in_features))

            else:
                self.scales.append(1.)

    def forward(self, inputs):
        s_out = []
        r_out = []
        u_out = []

        inputs = inputs.view(inputs.size(0), -1)
        for lif, ro, scale in zip(self.LIF_layers, self.readout_layers, self.scales):
            s, u = lif(inputs)
            r_ = ro(s) * scale

            inputs = s.detach()

            s_out.append(s)
            r_out.append(r_)
            u_out.append(u)

        return s_out, r_out, u_out
