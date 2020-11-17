import torch
from utils.loss import DECOLLELoss, one_hot_crossentropy
from model.LIF_MLP import LIFMLP
from tqdm import tqdm
from optimizer.BBSNN import BayesBiSNNRP
from copy import deepcopy
from utils.binarize import binarize, binarize_stochastic
import os
from torch.optim.lr_scheduler import StepLR
import argparse
import tables
import numpy as np
from data_preprocessing.load_data import get_batch_example
from collections import Counter
import pickle
import fnmatch
import time
from utils.train_utils import train_on_example_bbsnn
from utils.test_utils import mean_testing_dataset, mode_testing_dataset
from utils.misc import str2bool, get_acc

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects")
    parser.add_argument('--results', default=r"C:\Users\K1804053\results")
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--test_period', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--lr', type=float, default=1e4)
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
                             time.localtime()) + '_' + 'mnist_dvs_bbsnnrp' + r'_%d_epochs' % args.n_epochs\
               + '_temp_%3f' % args.temperature + '_prior_%3f' % args.prior_p + '_rho_%f' % args.rho + '_lr_%f' % args.lr
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
dt = 5000  # us
T = int(sample_length * 1000 / dt)  # number of timesteps in a sample
input_size = [676 * (1 + args.polarity)]
burnin = 100
args.labels = [i for i in range(10)]

dataset = tables.open_file(args.home + r'/datasets/mnist-dvs/mnist_dvs_events.hdf5')
train_data = dataset.root.train
test_data = dataset.root.test


n_examples_test = 1000
n_examples_train = 9000

binary_model = LIFMLP(input_size,
                      len(args.labels),
                      n_neurons=[512, 256],
                      with_output_layer=False,
                      with_bias=False,
                      prior_p=args.prior_p,
                      scaling=True,
                      softmax=args.with_softmax
                      ).to(args.device)

latent_model = deepcopy(binary_model)


# specify loss function
criterion = [torch.nn.SmoothL1Loss() for _ in range(binary_model.num_layers)]
if binary_model.with_output_layer:
    criterion[-1] = one_hot_crossentropy

decolle_loss = DECOLLELoss(criterion, latent_model)

# specify optimizer
optimizer = BayesBiSNNRP(binary_model.parameters(), latent_model.parameters(), lr=args.lr, temperature=args.temperature, prior_p=args.prior_p, rho=args.rho, device=args.device)

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
    print(acc)

    torch.save(binary_model.state_dict(), results_path + '/binary_model_weights.pt')
    torch.save(latent_model.state_dict(), results_path + '/latent_model_weights.pt')

    if (epoch + 1) % args.test_period == 0:
        binary_model.softmax = False

        ### Mode testing
        print('Mode testing on test data epoch %d/%d' % (epoch + 1, args.n_epochs))
        predictions_mode_test, idxs_test_mode = mode_testing_dataset(binary_model, optimizer, burnin, n_examples_test, args.batch_size,
                                                                     test_data, T, args.labels, input_size, dt, 26, args.polarity, args.device)
        np.save(os.path.join(results_path, 'test_predictions_latest_mode'), predictions_mode_test.numpy())
        np.save(os.path.join(results_path, 'idxs_test_mode'), np.array(idxs_test_mode))

        print('Mode testing on train data epoch %d/%d' % (epoch + 1, args.n_epochs))
        predictions_mode_train, idxs_train_mode = mode_testing_dataset(binary_model, optimizer, burnin, n_examples_train, args.batch_size,
                                                                       train_data, T, args.labels, input_size, dt, 26, args.polarity, args.device)
        np.save(os.path.join(results_path, 'train_predictions_latest_mode'), predictions_mode_train.numpy())
        np.save(os.path.join(results_path, 'idxs_train_mode'), np.array(idxs_train_mode))


        ### Mean testing
        print('Mean testing on test data epoch %d/%d' % (epoch + 1, args.n_epochs))
        predictions_mean_test, idxs_test_mean = mean_testing_dataset(binary_model, optimizer, burnin, args.n_samples, len(args.labels), n_examples_test,
                                                                     args.batch_size, test_data, T, args.labels, input_size, dt, 26, args.polarity, args.device)
        np.save(os.path.join(results_path, 'test_predictions_latest_mean'), predictions_mean_test.numpy())
        np.save(os.path.join(results_path, 'idxs_test_mean'), np.array(idxs_test_mean))

        print('Mean testing on train data epoch %d/%d' % (epoch + 1, args.n_epochs))
        predictions_mean_train, idxs_train_mean = mean_testing_dataset(binary_model, optimizer, burnin, args.n_samples, len(args.labels), n_examples_train,
                                                                       args.batch_size, train_data, T, args.labels, input_size, dt, 26, args.polarity, args.device)
        np.save(os.path.join(results_path, 'train_predictions_latest_mean'), predictions_mean_train.numpy())
        np.save(os.path.join(results_path, 'idxs_train_mean'), np.array(idxs_train_mean))
