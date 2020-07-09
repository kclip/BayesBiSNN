class LenetLIF(LIFNetwork):
    def __init__(self,
                 input_shape,
                 Nhid=[1],
                 Mhid=[128],
                 out_channels=1,
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2],
                 tau_mem=[10],
                 tau_syn=[6],
                 tau_ref=[2],
                 activation=smooth_step,
                 dropout=[0.5],
                 num_conv_layers=2,
                 num_mlp_layers=1,
                 lc_ampl=.5,
                 lif_layer_type=LIFLayer):

        num_layers = num_conv_layers + num_mlp_layers
        # If only one value provided, then it is duplicated for each layer
        if len(kernel_size) == 1:
            kernel_size = kernel_size * num_conv_layers
        if len(stride) == 1:
            stride = stride * num_conv_layers
        if len(pool_size) == 1:
            pool_size = pool_size * num_conv_layers
        if len(tau_mem) == 1:
            tau_mem = tau_mem * (num_layers + 1)
        if len(tau_syn) == 1:
            tau_syn = tau_syn * (num_layers + 1)
        if len(tau_ref) == 1:
            tau_ref = tau_ref * (num_layers + 1)
        if len(dropout) == 1:
            self.dropout = dropout = dropout * num_layers
        if len(Nhid) == 1:
            self.Nhid = Nhid = Nhid * num_conv_layers
        if len(Mhid) == 1:
            self.Mhid = Mhid = Mhid * num_mlp_layers

        super(LenetLIF, self).__init__()
        # Computing padding to preserve feature size
        padding = (np.array(kernel_size) - 1) // 2

        feature_height = input_shape[1]
        feature_width = input_shape[2]

        # THe following lists need to be nn.ModuleList in order for pytorch to properly load and save the state_dict
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.input_shape = input_shape
        Nhid = [input_shape[0]] + Nhid
        self.num_conv_layers = num_conv_layers
        self.num_mlp_layers = num_mlp_layers

        for i in range(num_conv_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width],
                kernel_size=kernel_size[i],
                stride=stride[i],
                padding=padding[i],
                dilation=1)
            feature_height //= pool_size[i]
            feature_width //= pool_size[i]
            base_layer = nn.Conv2d(Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i])
            layer = lif_layer_type(base_layer,
                                   alpha=alpha[i],
                                   beta=beta[i],
                                   alpharp=alpharp[i],
                                   deltat=deltat)
            pool = nn.MaxPool2d(kernel_size=pool_size[i])
            readout = nn.Linear(int(feature_height * feature_width * Nhid[i + 1]), out_channels)

            # Readout layer has random fixed weights
            for param in readout.parameters():
                param.requires_grad = False
            self.reset_lc_parameters(readout, lc_ampl)

            dropout_layer = nn.Dropout(dropout[i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(pool)
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

        mlp_in = int(feature_height * feature_width * Nhid[-1])
        Mhid = [mlp_in] + Mhid
        for i in range(num_mlp_layers):
            base_layer = nn.Linear(Mhid[i], Mhid[i + 1])
            layer = lif_layer_type(base_layer,
                                   alpha=alpha[i],
                                   beta=beta[i],
                                   alpharp=alpharp[i],
                                   deltat=deltat)
            readout = nn.Linear(Mhid[i + 1], out_channels)

            # Readout layer has random fixed weights
            for param in readout.parameters():
                param.requires_grad = False
            self.reset_lc_parameters(readout, lc_ampl)

            dropout_layer = nn.Dropout(dropout[self.num_conv_layers + i])

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
            self.readout_layers.append(readout)
            self.dropout_layers.append(dropout_layer)

    def forward(self, input):
        s_out = []
        r_out = []
        u_out = []
        i = 0
        for lif, pool, ro, do in zip(self.LIF_layers, self.pool_layers, self.readout_layers, self.dropout_layers):
            if i == self.num_conv_layers:
                input = input.view(input.size(0), -1)
            s, u = lif(input)
            u_p = pool(u)
            s_ = smooth_step(u_p)
            sd_ = do(s_)
            r_ = ro(sd_.reshape(sd_.size(0), -1))
            s_out.append(s_)
            r_out.append(r_)
            u_out.append(u_p)
            input = s_.detach()
            i += 1

        return s_out, r_out, u_out
