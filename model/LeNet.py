from model.LIF_base import *
from utils.activations import smooth_step
from torch.nn.init import _calculate_correct_fan, calculate_gain


class LenetLIF(LIFNetwork):
    def __init__(self,
                 input_shape,
                 Nhid_conv=[1],
                 Nhid_mlp=[128],
                 out_channels=1,
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 tau_mem=[10],
                 tau_syn=[6],
                 tau_ref=[2],
                 prior_p=0.5,
                 activation=smooth_step,
                 dropout=[0.5],
                 num_conv_layers=2,
                 num_mlp_layers=1,
                 lc_ampl=.5,
                 lif_layer_type=LIFLayer,
                 with_bias=True,
                 scaling=True,
                 with_output_layer=True):

        self.num_layers = num_conv_layers + num_mlp_layers
        self.activation = activation
        # If only one value provided, then it is duplicated for each layer
        if len(kernel_size) == 1:
            kernel_size = kernel_size * num_conv_layers
        if len(stride) == 1:
            stride = stride * num_conv_layers
        if len(pool_size) == 1:
            pool_size = pool_size * num_conv_layers
        if len(tau_mem) == 1:
            tau_mem = tau_mem * self.num_layers
        if len(tau_syn) == 1:
            tau_syn = tau_syn * self.num_layers
        if len(tau_ref) == 1:
            tau_ref = tau_ref * self.num_layers
        if len(dropout) == 1:
            dropout = dropout * self.num_layers
        if len(Nhid_conv) == 1:
            Nhid_conv = Nhid_conv * num_conv_layers
        if len(Nhid_mlp) == 1:
            Nhid_mlp = Nhid_mlp * num_mlp_layers

        super(LenetLIF, self).__init__()
        # Computing padding to preserve feature size
        padding = (np.array(kernel_size) - 1) // 2

        feature_height = input_shape[1]
        feature_width = input_shape[2]

        # THe following lists need to be nn.ModuleList in order for pytorch to properly load and save the state_dict
        self.scales = []
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.input_shape = input_shape
        Nhid_conv = [input_shape[0]] + Nhid_conv

        self.num_conv_layers = num_conv_layers
        self.num_mlp_layers = num_mlp_layers

        self.with_output_layer = with_output_layer

        for i in range(num_conv_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width],
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                dilation=1)
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]
            base_layer = nn.Conv2d(Nhid_conv[i], Nhid_conv[i + 1], kernel_size[i], stride[i], padding[i], bias=with_bias)
            base_layer.weight.data[:] = (2 * torch.bernoulli(torch.ones(base_layer.weight.shape) * prior_p) - 1) * 10  # / Mhid[i]
            if with_bias:
                base_layer.bias.data[:] = (2 * torch.bernoulli(torch.ones(base_layer.bias.shape) * prior_p) - 1) * 10

            layer = lif_layer_type(base_layer,
                                   activation=activation,
                                   tau_mem=tau_mem[i],
                                   tau_syn=tau_syn[i],
                                   tau_ref=tau_ref[i],
                                   scaling=scaling
                                   )

            pool = nn.MaxPool2d(kernel_size=pool_size[i])
            readout = nn.Linear(int(feature_height * feature_width * Nhid_conv[i + 1]), out_channels, bias=with_bias)
            readout.weight.data[:] = (2 * torch.bernoulli(torch.ones(readout.weight.shape) * prior_p) - 1)
            if with_bias:
                readout.bias.data[:] = (2 * torch.bernoulli(torch.ones(readout.bias.shape) * prior_p) - 1)

            # Readout layer has random fixed weights
            for param in readout.parameters():
                param.requires_grad = False

            dropout_layer = nn.Dropout(dropout[i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(pool)
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)
            if scaling and hasattr(readout, 'in_features'):
                fan = _calculate_correct_fan(readout.weight, mode='fan_in')
                gain = calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
                std = gain / math.sqrt(fan)
                self.scales.append(math.sqrt(3.0) * std)

                # self.scales.append(1. / np.prod(readout.in_features))
            else:
                self.scales.append(1.)


        mlp_in = int(feature_height * feature_width * Nhid_conv[-1])
        Nhid_mlp = [mlp_in] + Nhid_mlp
        for i in range(num_mlp_layers):
            base_layer = nn.Linear(Nhid_mlp[i], Nhid_mlp[i + 1])
            layer = lif_layer_type(base_layer,
                                   activation=activation,
                                   tau_mem=tau_mem[i],
                                   tau_syn=tau_syn[i],
                                   tau_ref=tau_ref[i]
                                   )

            if self.with_output_layer and (i+1 == self.num_mlp_layers):
                readout = nn.Identity()
                dropout_layer = nn.Identity()
                # layer.activation = torch.sigmoid
            else:
                readout = nn.Linear(Nhid_mlp[i + 1], out_channels)

                # Readout layer has random fixed weights
                for param in readout.parameters():
                    param.requires_grad = False

                dropout_layer = nn.Dropout(dropout[self.num_conv_layers + i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

            if scaling and hasattr(readout, 'in_features'):
                self.scales.append(1. / np.prod(readout.in_features))
            else:
                self.scales.append(1.)


    def forward(self, inputs):
        s_out = []
        r_out = []
        u_out = []
        i = 0
        for lif, pool, ro, do, scale in zip(self.LIF_layers, self.pool_layers, self.readout_layers, self.dropout_layers, self.scales):
            if i == self.num_conv_layers:
                inputs = inputs.view(inputs.size(0), -1)
            s, u = lif(inputs)
            u_p = pool(u)
            s_ = self.activation(u_p)
            sd_ = do(s_)
            r_ = ro(sd_.reshape(sd_.size(0), -1)) * scale
            s_out.append(s_)
            r_out.append(r_)
            u_out.append(u_p)
            inputs = s_.detach()
            i += 1

        return s_out, r_out, u_out
