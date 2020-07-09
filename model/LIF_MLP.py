from model.LIF_base import *
from utils.activations import smooth_step


class LIFMLP(LIFNetwork):
    def __init__(self,
                 input_shape,
                 output_shape,
                 n_neurons=[128],
                 tau_mem=[10],
                 tau_syn=[6],
                 tau_ref=[2],
                 activation=smooth_step,
                 num_layers=1,
                 lif_layer_type=LIFLayer,
                 lc_ampl=.5,
                 method='rtrl'):

        num_layers = max(num_layers, max([len(n_neurons), len(tau_mem), len(tau_syn), len(tau_ref)]))
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
            base_layer = nn.Linear(Mhid[i], Mhid[i+1])
            layer = lif_layer_type(base_layer,
                                   activation=activation,
                                   tau_mem=tau_mem[i],
                                   tau_syn=tau_syn[i],
                                   tau_ref=tau_ref[i],
                                   do_detach=True if method == 'rtrl' else False)


            readout = nn.Linear(Mhid[i+1], output_shape)
            for param in readout.parameters():
                param.requires_grad = False
            self.reset_lc_parameters(readout, lc_ampl)

            self.LIF_layers.append(layer)
            self.readout_layers.append(readout)

        self.activation = activation

    def forward(self, inputs):
        s_out = []
        r_out = []
        u_out = []

        inputs = inputs.view(inputs.size(0), -1)
        for lif, ro in zip(self.LIF_layers, self.readout_layers):
            s, u = lif(inputs)
            s_ = self.activation(u)
            r_ = ro(s_)

            inputs = s_.detach() if lif.do_detach else s_

            s_out.append(s_)
            r_out.append(r_)
            u_out.append(u)

        return s_out, r_out, u_out
