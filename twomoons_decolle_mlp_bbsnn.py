import torch
from utils.loss import DECOLLELoss, one_hot_crossentropy
from model.LIF_MLP import LIFMLP
from tqdm import tqdm
from optimizer.BBSNN import BayesBiSNNRP
from copy import deepcopy
import os
import argparse
import numpy as np
from data_preprocessing.load_data import get_batch_example
from collections import Counter
import pickle
import fnmatch
import time
from utils.misc import make_moon_dataset_bin, make_moon_dataset_bin_pop_coding, make_moon_test_dataset_bin_pop_coding
from utils.test_utils import mean_testing, mode_testing
from utils.train_utils import train_on_example_bbsnn
from utils.misc import str2bool, get_acc

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects")
    parser.add_argument('--results', default=r"C:\Users\K1804053\results")
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--test_period', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--rho', type=float, default=5e-4)
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
                             time.localtime()) + '_' + 'two_moons_mlp_bbsnnrp' + r'_%d_epochs' % args.n_epochs\
               + '_temp_%3f' % args.temperature + '_prior_%3f' % args.prior_p + '_rho_%f' % args.rho + '_lr_%f' % args.lr + '_softmax_' + args.with_softmax
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


T = 100
n_examples_train = 200
n_samples_per_dim_test = 200
n_examples_test = n_samples_per_dim_test ** 2
n_neurons_per_dim = 10
n_samples = 10
burnin = 10

x_bin_train, y_bin_train, x_train, y_train = make_moon_dataset_bin_pop_coding(n_examples_train, T, 0.1, n_neurons_per_dim)
np.save(os.path.join(results_path, 'x_train'), x_train)
np.save(os.path.join(results_path, 'y_train'), y_train)

x_bin_test, x_test, y_test = make_moon_test_dataset_bin_pop_coding(n_samples_per_dim_test, T, n_neurons_per_dim)
np.save(os.path.join(results_path, 'x_test'), x_test)
np.save(os.path.join(results_path, 'y_test'), y_test)

input_size = [x_bin_train.shape[-1]]

binary_model = LIFMLP(input_size,
                      2,
                      n_neurons=[64, 64],
                      with_output_layer=False,
                      with_bias=False,
                      prior_p=args.prior_p,
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

    n_batchs = n_examples_train // args.batch_size + (1 - (n_examples_train % args.batch_size == 0))
    idx_avail = np.arange(n_examples_train)
    idxs_used = []

    print('Epoch %d/%d' % (epoch + 1, args.n_epochs))
    preds = torch.FloatTensor()
    true_labels = torch.FloatTensor()

    for i in tqdm(range(n_batchs)):
        if (i == (n_batchs - 1)) & (n_examples_train % args.batch_size != 0):
            batch_size_curr = n_examples_train % args.batch_size
        else:
            batch_size_curr = args.batch_size


        idxs = np.random.choice(idx_avail, [batch_size_curr], replace=False)
        idxs_used += list(idxs)
        idx_avail = [i for i in idx_avail if i not in idxs_used]

        inputs = x_bin_train[idxs].transpose(0, 1).to(args.device)
        labels = y_bin_train[idxs].transpose(1, 2).to(args.device)

        optimizer.update_concrete_weights()
        binary_model.init(inputs, burnin=burnin)

        readout_hist = train_on_example_bbsnn(binary_model, optimizer, decolle_loss, inputs, labels, burnin, T)

        with torch.no_grad():
            preds = torch.cat((preds, torch.sum(readout_hist[-1].type_as(preds), dim=0)))
            true_labels = torch.cat((true_labels, labels.cpu().type_as(true_labels)))

    acc = get_acc(preds.argmax(dim=-1), true_labels, len(preds))
    print(acc)

    torch.save(binary_model.state_dict(), results_path + '/binary_model_weights.pt')
    torch.save(latent_model.state_dict(), results_path + '/latent_model_weights.pt')


    if (epoch + 1) % (args.n_epochs//5) == 0:
        torch.save(binary_model.state_dict(), results_path + '/binary_model_weights_%d.pt' % (1 + epoch))
        torch.save(latent_model.state_dict(), results_path + '/latent_model_weights_%d.pt' % (1 + epoch))


    if (epoch + 1) % args.test_period == 0:
        binary_model.softmax = False

        print('Mode testing on test data epoch %d/%d' % (epoch + 1, args.n_epochs))
        predictions_mode_test, idxs_test_mode = mode_testing(binary_model, optimizer, burnin, n_examples_test, args.batch_size, x_bin_test, T, args.device)
        np.save(os.path.join(results_path, 'test_predictions_latest_mode'), predictions_mode_test.numpy())
        np.save(os.path.join(results_path, 'idxs_test_mode'), np.array(idxs_test_mode))

        print('Mode testing on train data epoch %d/%d' % (epoch + 1, args.n_epochs))
        predictions_mode_train, idxs_train_mode = mode_testing(binary_model, optimizer, burnin, n_examples_train, args.batch_size, x_bin_train, T, args.device)
        np.save(os.path.join(results_path, 'train_predictions_latest_mode'), predictions_mode_train.numpy())
        np.save(os.path.join(results_path, 'idxs_train_mode'), np.array(idxs_train_mode))


        ### Mean testing
        print('Mean testing on test data epoch %d/%d' % (epoch + 1, args.n_epochs))
        predictions_mean_test, idxs_test_mean = mean_testing(binary_model, optimizer, burnin, n_samples, 2, n_examples_test, args.batch_size, x_bin_test, T, args.device)
        np.save(os.path.join(results_path, 'test_predictions_latest_mean'), predictions_mean_test.numpy())
        np.save(os.path.join(results_path, 'idxs_test_mean'), np.array(idxs_test_mean))

        print('Mean testing on train data epoch %d/%d' % (epoch + 1, args.n_epochs))
        predictions_mean_train, idxs_train_mean = mean_testing(binary_model, optimizer, burnin, n_samples, 2, n_examples_train, args.batch_size, x_bin_train, T, args.device)
        np.save(os.path.join(results_path, 'train_predictions_latest_mean'), predictions_mean_train.numpy())
        np.save(os.path.join(results_path, 'idxs_train_mean'), np.array(idxs_train_mean))

