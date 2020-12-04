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
from snn.utils.misc import find_indices_for_labels
from data_preprocessing.load_data import create_dataloader

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects")
    parser.add_argument('--results', default=r"C:\Users\K1804053\results")
    parser.add_argument('--dataset', default=r"mnist_dvs")
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--n_epochs', type=int, default=20000)
    parser.add_argument('--test_period', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=10)

    parser.add_argument('--lr', type=float, default=100)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--rho', type=float, default=1e-7)
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
                             time.localtime()) + '_' + args.dataset + '_bbsnnrp_lenet_' + r'_%d_epochs' % args.n_epochs \
               + '_temp_%3f' % args.temperature + '_prior_%3f' % args.prior_p + '_rho_%f' % args.rho + '_lr_%f' % args.lr
os.makedirs(results_path)

args.polarity = str2bool(args.polarity)
args.with_softmax = str2bool(args.with_softmax)
args.disable_cuda = str2bool(args.disable_cuda)

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

sample_length = 2e6  # length of samples during training in mus
dt = 1000  # us
T = int(sample_length / dt)
burnin = 100


if args.dataset == 'mnist_dvs':
    dataset_path = args.home + r'/datasets/mnist-dvs/mnist_dvs_events_new.hdf5'
elif args.dataset == 'dvs_gestures':
    dataset_path = args.home + r'/datasets/DvsGesture/dvs_gestures_events_new.hdf5'

dataset = tables.open_file(dataset_path)
train_data = dataset.root.train
test_data = dataset.root.test

args.classes = [i for i in range(dataset.root.stats.train_label[1])]

x_max = dataset.root.stats.train_data[1]
input_size = [2, x_max, x_max]
dataset.close()

train_dl, test_dl = create_dataloader(dataset_path, batch_size=args.batch_size, size=input_size, classes=args.classes, sample_length_train=sample_length,
                                      sample_length_test=sample_length, dt=dt, polarity=args.polarity, num_workers=2)
train_iterator = iter(train_dl)
test_iterator = iter(test_dl)


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
# criterion = [torch.nn.SmoothL1Loss() for _ in range(binary_model.num_layers)]
criterion = [one_hot_crossentropy for _ in range(binary_model.num_layers)]

if binary_model.with_output_layer:
    criterion[-1] = one_hot_crossentropy

decolle_loss = DECOLLELoss(criterion, latent_model)

# specify optimizer
optimizer = BayesBiSNNRP(binary_model.parameters(), latent_model.parameters(), lr=args.lr, temperature=args.temperature, prior_p=args.prior_p, rho=args.rho, device=args.device)

binary_model.init_parameters()

print([layer.scale for layer in binary_model.LIF_layers])
print(binary_model.scales)

for epoch in range(args.n_epochs):
    binary_model.softmax = args.with_softmax
    loss = 0

    inputs, labels = next(train_iterator)
    inputs = inputs.transpose(0, 1).to(args.device)
    labels = labels.to(args.device)

    optimizer.update_concrete_weights()
    binary_model.init(inputs, burnin=burnin)

    readout_hist = train_on_example_bbsnn(binary_model, optimizer, decolle_loss, inputs, labels, burnin, T)
    acc = get_acc(torch.sum(readout_hist[-1], dim=0).argmax(dim=1), labels, args.batch_size)
    print(acc)
    acc = get_acc(torch.sum(readout_hist[-2], dim=0).argmax(dim=1), labels, args.batch_size)
    print(acc)


    torch.save(binary_model.state_dict(), results_path + '/binary_model_weights.pt')
    torch.save(latent_model.state_dict(), results_path + '/latent_model_weights.pt')


    if (epoch + 1) % (args.n_epochs//5) == 0:
        torch.save(binary_model.state_dict(), results_path + '/binary_model_weights_%d.pt' % (1 + epoch))
        torch.save(latent_model.state_dict(), results_path + '/latent_model_weights_%d.pt' % (1 + epoch))


    if (epoch + 1) % args.test_period == 0:
        binary_model.softmax = False
        launch_tests(binary_model, optimizer, burnin, None, test_iterator, T, epoch, args, results_path, output=-1)
