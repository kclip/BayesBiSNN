import torch
from utils.loss import DECOLLELoss, one_hot_crossentropy
from model.LIF_MLP import LIFMLP
from tqdm import tqdm
from optimizer.BBSNN import BayesBiSNNRP
from copy import deepcopy
import os
import argparse
import numpy as np
from data_preprocessing.load_data_old import get_batch_example
from collections import Counter
import pickle
import fnmatch
import time
from utils.misc import make_moon_dataset_bin, make_moon_dataset_bin_pop_coding, make_moon_test_dataset_bin_pop_coding

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects")
    parser.add_argument('--results', default=r"C:\Users\K1804053\results")
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--n_epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--rho', type=float, default=5e-4)
    parser.add_argument('--prior_p', type=float, default=0.5)
    parser.add_argument('--with_softmax', type=str, default='true')
    parser.add_argument('--disable-cuda', type=str, default='false', help='Disable CUDA')

    args = parser.parse_args()

args.with_softmax = str2bool(args.with_softmax)
args.disable_cuda = str2bool(args.disable_cuda)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

T = 100
n_samples_train = 200
n_samples_per_dim_test = 100
n_samples_test = n_samples_per_dim_test ** 2
n_neurons_per_dim = 10

x_bin_test, x_test, y_test = make_moon_test_dataset_bin_pop_coding(n_samples_per_dim_test, T, n_neurons_per_dim)
np.save(os.path.join(args.results, 'x_test'), x_test)
np.save(os.path.join(args.results, 'y_test'), y_test)


batch_size = 32
input_size = [x_bin_test.shape[-1]]
burnin = 10

binary_model = LIFMLP(input_size,
                      2,
                      n_neurons=[64, 64],
                      with_output_layer=False,
                      with_bias=False,
                      prior_p=args.prior_p,
                      softmax=args.with_softmax
                      ).to(args.device)


latent_model = deepcopy(binary_model)

# specify optimizer
binary_model.load_state_dict(torch.load(args.results + '/binary_model_weights.pt'))

optimizer = BayesBiSNNRP(binary_model.parameters(), latent_model.parameters(), lr=args.lr, temperature=args.temperature, prior_p=args.prior_p, rho=args.rho, device=args.device)

binary_model.softmax = False

with torch.no_grad():
    n_batchs_test = n_samples_test // batch_size + (1 - (n_samples_test % batch_size == 0))
    idx_avail_test = np.arange(n_samples_test)
    idxs_used_test = []

    print('Mode testing epoch')
    predictions = torch.FloatTensor()

    for i in tqdm(range(n_batchs_test)):
        if (i == (n_batchs_test - 1)) & (n_samples_test % batch_size != 0):
            batch_size_curr = n_samples_test % batch_size
        else:
            batch_size_curr = batch_size

        idxs_test = np.random.choice(idx_avail_test, [batch_size_curr], replace=False)
        idxs_used_test += list(idxs_test)
        idx_avail_test = [i for i in idx_avail_test if i not in idxs_used_test]

        inputs = x_bin_test[idxs_test].permute(1, 0, 2).to(args.device)

        binary_model.init(inputs, burnin=burnin)

        readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

        for t in range(burnin, T):
            # forward pass: compute predicted outputs by passing inputs to the model
            s, r, u = binary_model(inputs[t])

            for l, ro_h in enumerate(readout_hist):
                readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

        predictions = torch.cat((predictions, torch.sum(readout_hist[-1], dim=0)))

    np.save(os.path.join(args.results, 'test_predictions_latest'), predictions.numpy())
    np.save(os.path.join(args.results, 'idxs_test'), np.array(idxs_used_test))

    n_batchs_train = n_samples_train // batch_size + (1 - (n_samples_train % batch_size == 0))
    idx_avail_train = np.arange(n_samples_train)
    idxs_used_train = []