from model.LIF_base import *
from utils.activations import smooth_step
from utils.misc import get_output_shape, get_scale

'''
LIF SNN with local errors and LeNet architecture
Adapted from https://github.com/nmi-lab/decolle-public
'''

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
                 lif_layer_type=LIFLayer,
                 with_bias=True,
                 with_readout=True,
                 scaling=True,
                 softmax=True):

        self.softmax = softmax  # Apply softmax to outputs of readout layers
        self.num_layers = num_conv_layers + num_mlp_layers

        self.activation = activation # Activation function of LIF neurons

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

        self.scales = []  # Downscaling of the readout outputs
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.input_shape = input_shape
        Nhid_conv = [input_shape[0]] + Nhid_conv

        self.num_conv_layers = num_conv_layers
        self.num_mlp_layers = num_mlp_layers

        # Creates LIF convolutional layers
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

            # Initialize weights in {-5, 5} with probas following a Bernoulli distribution and probability prior_
            base_layer.weight.data[:] = (2 * torch.bernoulli(torch.ones(base_layer.weight.shape) * prior_p) - 1) * 5
            if with_bias:
                base_layer.bias.data[:] = (2 * torch.bernoulli(torch.ones(base_layer.bias.shape) * prior_p) - 1) * 5

            layer = lif_layer_type(base_layer,
                                   activation=activation,
                                   tau_mem=tau_mem[i],
                                   tau_syn=tau_syn[i],
                                   tau_ref=tau_ref[i],
                                   scaling=scaling
                                   )

            pool = nn.MaxPool2d(kernel_size=pool_size[i])

            if with_readout:
                # Initialize readout weights in {-1, 1} with probas following a Bernoulli distribution and probability prior_p
                readout = nn.Linear(int(feature_height * feature_width * Nhid_conv[i + 1]), out_channels, bias=with_bias)
                readout.weight.data[:] = (2 * torch.bernoulli(torch.ones(readout.weight.shape) * prior_p) - 1)
                if with_bias:
                    readout.bias.data[:] = (2 * torch.bernoulli(torch.ones(readout.bias.shape) * prior_p) - 1)
            else:
                readout = nn.Identity()

            for param in readout.parameters():
                param.requires_grad = False


            dropout_layer = nn.Dropout(dropout[i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(pool)
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

            self.scales.append(get_scale(readout, scaling))


        mlp_in = int(feature_height * feature_width * Nhid_conv[-1])
        Nhid_mlp = [mlp_in] + Nhid_mlp

        # Creates LIF linear layers
        for i in range(num_mlp_layers):
            base_layer = nn.Linear(Nhid_mlp[i], Nhid_mlp[i + 1])

            # Initialize weights in {-5, 5} with probas following a Bernoulli distribution and probability prior_
            base_layer.weight.data[:] = (2 * torch.bernoulli(torch.ones(base_layer.weight.shape) * prior_p) - 1) * 5
            if with_bias:
                base_layer.bias.data[:] = (2 * torch.bernoulli(torch.ones(base_layer.bias.shape) * prior_p) - 1) * 5
            layer = lif_layer_type(base_layer,
                                   activation=activation,
                                   tau_mem=tau_mem[i],
                                   tau_syn=tau_syn[i],
                                   tau_ref=tau_ref[i]
                                   )

            if with_readout:
                # Initialize readout weights in {-1, 1} with probas following a Bernoulli distribution and probability prior_p
                readout = nn.Linear(Nhid_mlp[i + 1], out_channels)
                readout.weight.data[:] = (2 * torch.bernoulli(torch.ones(readout.weight.shape) * prior_p) - 1)
                if with_bias:
                    readout.bias.data[:] = (2 * torch.bernoulli(torch.ones(readout.bias.shape) * prior_p) - 1)
                # Readout layer has random fixed weights
                for param in readout.parameters():
                    param.requires_grad = False
            else:
                readout = nn.Identity()


            dropout_layer = nn.Dropout(dropout[self.num_conv_layers + i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

            self.scales.append(get_scale(readout, scaling))


    def forward(self, inputs):
        s_out = []
        r_out = []
        u_out = []
        i = 0

        # Forward propagates the signal through all layers
        for lif, pool, ro, do, scale in zip(self.LIF_layers, self.pool_layers, self.readout_layers, self.dropout_layers, self.scales):
            if i == self.num_conv_layers:
                inputs = inputs.view(inputs.size(0), -1)
            s, u = lif(inputs)
            u_p = pool(u)
            s_ = self.activation(u_p)
            sd_ = do(s_)
            r_ = ro(sd_.reshape(sd_.size(0), -1)) * scale  # Readout outputs are scaled down
            s_out.append(s_)
            if self.softmax:
                r_ = torch.softmax(r_, dim=-1)
            r_out.append(r_)
            u_out.append(u_p)
            inputs = s_.detach()  # Gradients are not propagated through the layers
            i += 1
        return s_out, r_out, u_out
