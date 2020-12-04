import torch
from utils.loss import DECOLLELoss, one_hot_crossentropy
from model.LeNet import LenetLIF
from tqdm import tqdm
from optimizer.BBSNN import BayesBiSNNRP
from copy import deepcopy
from utils.binarize import binarize, binarize_stochastic
import os
from torch.optim.lr_scheduler import StepLR
import argparse
import tables
import numpy as np
from data_preprocessing.load_data_old import get_batch_example
from collections import Counter
import pickle
import fnmatch
import time
from utils.train_utils import train_on_example_bbsnn
from utils.test_utils import launch_tests
from utils.misc import str2bool, get_acc


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects")
    parser.add_argument('--results', default=r"C:\Users\K1804053\results")
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--prior_p', type=float, default=0.5)
    parser.add_argument('--with_softmax', type=str, default='true')
    parser.add_argument('--polarity', type=str, default='true')
    parser.add_argument('--disable-cuda', type=str, default='false', help='Disable CUDA')

    args = parser.parse_args()

prelist = np.sort(fnmatch.filter(os.listdir(args.results), '[0-9][0-9][0-9]__*'))
if len(prelist) == 0:
    expDirN = "001"
else:
    expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

results_path = time.strftime(args.results + r'/' + expDirN + "__" + "%d-%m-%Y",
                             time.localtime()) + '_' + 'search_mnist_dvs_bbsnnrp_lenet_' + r'_%d_epochs' % args.n_epochs \
               + '_temp_%3f' % args.temperature + '_prior_%3f' % args.prior_p
os.makedirs(results_path)

args.polarity = str2bool(args.polarity)
args.with_softmax = str2bool(args.with_softmax)
args.disable_cuda = str2bool(args.disable_cuda)

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.train_accs = {i: [] for i in range(0, args.n_epochs, 100)}
args.train_accs[args.n_epochs] = []

sample_length = 2000  # length of samples during training in ms
dt = 1000  # us
T = int(sample_length * 1000 / dt)  # number of timesteps in a sample
input_size = [2, 26, 26]
burnin = 100
n_examples_test = 1000
n_examples_train = 9000

args.labels = [i for i in range(10)]

dataset = tables.open_file(args.home + r'/datasets/mnist-dvs/mnist_dvs_events.hdf5')
train_data = dataset.root.train
test_data = dataset.root.test


lr_list = np.logspace(1, 6, 6, endpoint=True)
rho_list = np.logspace(-14, -3, 12, endpoint=True)

results_l1 = {i: {j: [] for j in rho_list} for i in lr_list}
results_l2 = {i: {j: [] for j in rho_list} for i in lr_list}

for lr in lr_list:
    for rho in rho_list:
        print('LR: ' + str(lr) + ', rho: ' + str(rho))
        binary_model = LenetLIF(input_size,
                                Nhid_conv=[64, 128, 128],
                                Nhid_mlp=[],
                                out_channels=10,
                                kernel_size=[7],
                                stride=[1],
                                pool_size=[2, 1, 2],
                                dropout=[0.],
                                num_conv_layers=3,
                                num_mlp_layers=0,
                                with_bias=True,
                                with_output_layer=False,
                                softmax=args.with_softmax).to(args.device)
        latent_model = deepcopy(binary_model)

        # specify loss function
        criterion = [one_hot_crossentropy for _ in range(binary_model.num_layers)]

        decolle_loss = DECOLLELoss(criterion, latent_model)

        # specify optimizer
        optimizer = BayesBiSNNRP(binary_model.parameters(), latent_model.parameters(), lr=lr, temperature=args.temperature, prior_p=args.prior_p, rho=rho, device=args.device)

        binary_model.init_parameters()

        for epoch in range(args.n_epochs):
            binary_model.softmax = args.with_softmax
            loss = 0

            idxs = np.random.choice(np.arange(n_examples_train), [args.batch_size], replace=False)

            inputs, labels = get_batch_example(train_data, idxs, args.batch_size, T, args.labels, input_size, dt, 26, args.polarity)

            inputs = inputs.transpose(0, 1).to(args.device)
            labels = labels.to(args.device)

            optimizer.update_concrete_weights()
            binary_model.init(inputs, burnin=burnin)

            readout_hist = train_on_example_bbsnn(binary_model, optimizer, decolle_loss, inputs, labels, burnin, T)
            acc = get_acc(torch.sum(readout_hist[-1], dim=0).argmax(dim=1), labels, args.batch_size)
            results_l1[lr][rho].append(acc)
            print(acc)
            acc = get_acc(torch.sum(readout_hist[-2], dim=0).argmax(dim=1), labels, args.batch_size)
            print(acc)
            results_l2[lr][rho].append(acc)
