import torch
from utils.loss import DECOLLELoss, one_hot_crossentropy
from model.LIF_MLP import LIFMLP
from tqdm import tqdm
from optimizer.BiSGD import BiSGD
from copy import deepcopy
import os
import argparse
import numpy as np
from data_preprocessing.load_data_old import get_batch_example
from collections import Counter
import pickle
import fnmatch
import time
from utils.misc import gen_1d_signal_realtarget
from utils.test_utils import mean_testing, mode_testing
from utils.train_utils import train_on_example_bbsnn
from utils.misc import str2bool, get_acc
from utils.binarize import binarize

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects")
    parser.add_argument('--results', default=r"C:\Users\K1804053\results")
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--test_period', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)

    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--disable-cuda', type=str, default='false', help='Disable CUDA')

    args = parser.parse_args()

prelist = np.sort(fnmatch.filter(os.listdir(args.results), '[0-9][0-9][0-9]__*'))
if len(prelist) == 0:
    expDirN = "001"
else:
    expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

results_path = time.strftime(args.results + r'/' + expDirN + "__" + "%d-%m-%Y",
                             time.localtime()) + '_' + '1d_mlp_stlr' + r'_%d_epochs' % args.n_epochs + '_lr_%f' % args.lr
os.makedirs(results_path)

args.disable_cuda = str2bool(args.disable_cuda)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.train_accs = {i: [] for i in range(0, args.n_epochs, 100)}
args.train_accs[args.n_epochs] = []


T = 100
n_neurons_per_dim = 20
step = 100

x_train, y_train, x_test, y_test, x_train_bin, x_test_bin = gen_1d_signal_realtarget(T=T, step_train=20, step_test=step, n_neuron_per_dim=n_neurons_per_dim)
n_examples_train = len(x_train)
n_examples_test = len(x_test)
n_samples = 10

np.save(os.path.join(results_path, 'x_train'), x_train)
np.save(os.path.join(results_path, 'y_train'), y_train)
np.save(os.path.join(results_path, 'x_test'), x_test)
np.save(os.path.join(results_path, 'y_test'), y_test)


input_size = [x_train_bin.shape[-1]]
burnin = 10

binary_model = LIFMLP(input_size,
                      1,
                      n_neurons=[128, 128],
                      with_output_layer=False,
                      with_bias=False,
                      prior_p=0.5,
                      softmax=False
                      ).to(args.device)

latent_model = deepcopy(binary_model)

# specify loss function
criterion = [torch.nn.SmoothL1Loss() for _ in range(binary_model.num_layers)]
if binary_model.with_output_layer:
    criterion[-1] = one_hot_crossentropy

decolle_loss = DECOLLELoss(criterion, latent_model)

# specify optimizer
optimizer = BiSGD(binary_model.parameters(), latent_model.parameters(), lr=args.lr, binarizer=binarize)

binary_model.init_parameters()

for epoch in range(args.n_epochs):
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

        inputs = x_train_bin[idxs].transpose(0, 1).to(args.device)
        labels = torch.FloatTensor(y_train[idxs]).unsqueeze(1).unsqueeze(2).repeat(1, 1, T).to(args.device)

        binary_model.init(inputs, burnin=burnin)

        readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

        for t in range(burnin, T):
            # print(list(binary_model.parameters()))

            # forward pass: compute predicted outputs by passing inputs to the model
            s, r, u = binary_model(inputs[t])

            for l, ro_h in enumerate(readout_hist):
                readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

            # calculate the loss
            loss = decolle_loss(s, r, u, target=labels[:, :, t])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    torch.save(binary_model.state_dict(), results_path + '/binary_model_weights.pt')
    torch.save(latent_model.state_dict(), results_path + '/latent_model_weights.pt')


    if (epoch + 1) % (args.n_epochs//5) == 0:
        torch.save(binary_model.state_dict(), results_path + '/binary_model_weights_%d.pt' % (1 + epoch))
        torch.save(latent_model.state_dict(), results_path + '/latent_model_weights_%d.pt' % (1 + epoch))

    if (epoch + 1) % args.test_period == 0:
        binary_model.softmax = False
        ### Mode testing
        with torch.no_grad():
            n_batchs_test = n_examples_test // args.batch_size + (1 - (n_examples_test % args.batch_size == 0))
            idx_avail_test = np.arange(n_examples_test)
            idxs_used_test = []

            print('Testing epoch %d/%d' % (epoch + 1, args.n_epochs))
            predictions = torch.FloatTensor()

            for i in tqdm(range(n_batchs_test)):
                if (i == (n_batchs_test - 1)) & (n_examples_test % args.batch_size != 0):
                    batch_size_curr = n_examples_test % args.batch_size
                else:
                    batch_size_curr = args.batch_size

                idxs_test = np.random.choice(idx_avail_test, [batch_size_curr], replace=False)
                idxs_used_test += list(idxs_test)
                idx_avail_test = [i for i in idx_avail_test if i not in idxs_used_test]

                inputs = x_test_bin[idxs_test].permute(1, 0, 2).to(args.device)

                binary_model.init(inputs, burnin=burnin)

                readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

                for t in range(burnin, T):
                    # forward pass: compute predicted outputs by passing inputs to the model
                    s, r, u = binary_model(inputs[t])

                    for l, ro_h in enumerate(readout_hist):
                        readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

                predictions = torch.cat((predictions, torch.sum(readout_hist[-1], dim=0)))

            np.save(os.path.join(results_path, 'test_predictions_latest'), predictions.numpy())
            np.save(os.path.join(results_path, 'idxs_test'), np.array(idxs_used_test))

            n_batchs_train = n_examples_train // args.batch_size + (1 - (n_examples_train % args.batch_size == 0))
            idx_avail_train = np.arange(n_examples_train)
            idxs_used_train = []
