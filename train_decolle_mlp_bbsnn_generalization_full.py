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
from snn.utils.misc import find_indices_for_labels

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
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--rho', type=float, default=1e-4)
    parser.add_argument('--prior_p', type=float, default=0.5)
    parser.add_argument('--disable-cuda', type=str, default='false', help='Disable CUDA')
    parser.add_argument('--with_softmax', type=str, default='true')
    parser.add_argument('--labels_train', nargs='+', default=None, type=int, help='Class labels to be used during training')

    args = parser.parse_args()

prelist = np.sort(fnmatch.filter(os.listdir(args.results), '[0-9][0-9][0-9]__*'))
if len(prelist) == 0:
    expDirN = "001"
else:
    expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

results_path = time.strftime(args.results + r'/' + expDirN + "__" + "%d-%m-%Y",
                             time.localtime()) + '_' + 'mnist_dvs_bbsnnrp_genfull' + r'_%d_epochs' % args.n_epochs\
               + '_temp_%3f' % args.temperature + '_prior_%3f' % args.prior_p + '_rho_%f' % args.rho + '_lr_%f' % args.lr
os.makedirs(results_path)

args.with_softmax = str2bool(args.with_softmax)
args.disable_cuda = str2bool(args.disable_cuda)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

test_period = 2000
batch_size = 32
sample_length = 2000  # length of samples during training in ms
dt = 5000  # us
T = int(sample_length * 1000 / dt)  # number of timesteps in a sample
input_size = [676 * 2]
burnin = 100

try:
    dataset = tables.open_file(args.home + r'/datasets/mnist-dvs/mnist_dvs_events.hdf5')
except:
    dataset = tables.open_file(r'/scratch/users/k1804053/datasets/mnist-dvs/mnist_dvs_events.hdf5')

train_data = dataset.root.test
test_data = dataset.root.train

args.labels_test = [i for i in range(10) if i not in args.labels_train]

samples_train = find_indices_for_labels(train_data, args.labels_train)
n_samples_train = len(samples_train)

samples_val = find_indices_for_labels(test_data, args.labels_train)
n_samples_val = len(samples_val)

samples_test = find_indices_for_labels(test_data, args.labels_test)
n_samples_test = len(samples_test)


binary_model = LIFMLP(input_size,
                      len(args.labels_train),
                      n_neurons=[512, 256],
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

batch_size_val = batch_size // 10
batch_size_train = batch_size - batch_size_val

for epoch in range(args.n_epochs):
    binary_model.softmax = args.with_softmax

    loss = 0

    idxs_train = np.random.choice(samples_train, [batch_size_train], replace=False)
    idxs_val = np.random.choice(samples_val, [batch_size_val], replace=False)

    inputs_train, labels_train = get_batch_example(train_data, idxs_train, batch_size_train, T, args.labels_train, input_size, dt, 26, True)
    inputs_val, labels_val = get_batch_example(test_data, idxs_val, batch_size_val, T, args.labels_train, input_size, dt, 26, True)

    inputs = torch.cat((inputs_train.permute(1, 0, 2), inputs_val.permute(1, 0, 2)), dim=1).to(args.device)
    labels = torch.cat((labels_train, labels_val), dim=0).to(args.device)

    optimizer.update_concrete_weights()

    binary_model.init(inputs, burnin=burnin)

    readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

    for t in range(burnin, T):
        # forward pass: compute new pseudo-binary weights
        optimizer.update_concrete_weights()
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


    if (epoch + 1) % (args.n_epochs//5) == 0:
        torch.save(binary_model.state_dict(), results_path + '/binary_model_weights_%d.pt' % (1 + epoch))
        torch.save(latent_model.state_dict(), results_path + '/latent_model_weights_%d.pt' % (1 + epoch))


    if (epoch + 1) % test_period == 0:
        binary_model.softmax = False

        ### Mode testing
        with torch.no_grad():
            # # Compute weights
            optimizer.get_concrete_weights_mode()

            n_batchs_test = n_samples_test // batch_size + (1 - (n_samples_test % batch_size == 0))
            idx_avail_test = samples_test
            idxs_used_test_mode = []

            print('Mode testing on test data epoch %d/%d' % (epoch + 1, args.n_epochs))
            predictions_mode = torch.FloatTensor()
            labels_mode = torch.FloatTensor()

            for i in range(n_batchs_test):
                if (i == (n_batchs_test - 1)) & (n_samples_test % batch_size != 0):
                    batch_size_curr = n_samples_test % batch_size
                else:
                    batch_size_curr = batch_size

                idxs_test = np.random.choice(idx_avail_test, [batch_size_curr], replace=False)
                idxs_used_test_mode += list(idxs_test)
                idx_avail_test = [i for i in idx_avail_test if i not in idxs_used_test_mode]

                inputs, _ = get_batch_example(test_data, idxs_test, batch_size_curr, T, args.labels_test, input_size, dt, 26, True)
                inputs = inputs.permute(1, 0, 2).to(args.device)

                binary_model.init(inputs, burnin=burnin)

                readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

                for t in range(burnin, T):
                    # forward pass: compute predicted outputs by passing inputs to the model
                    s, r, u = binary_model(inputs[t])

                    for l, ro_h in enumerate(readout_hist):
                        readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

                predictions_mode = torch.cat((predictions_mode, readout_hist[-1]))

            np.save(os.path.join(results_path, 'test_predictions_latest_mode'), predictions_mode.numpy())
            np.save(os.path.join(results_path, 'idxs_test_mode'), np.array(idxs_used_test_mode))


            ### Mode testing on train data
            n_batchs = n_samples_train // batch_size + (1 - (n_samples_train % batch_size == 0))
            idx_avail = samples_train
            idxs_used_train_mode = []
            preds = torch.FloatTensor()
            labels_mode = torch.FloatTensor()

            print('Mode testing on train data epoch %d/%d' % (epoch + 1, args.n_epochs))
            # for i in tqdm(range(n_batchs)):
            for i in range(n_batchs):
                if (i == (n_batchs - 1)) & (n_samples_train % batch_size != 0):
                    batch_size_curr = n_samples_train % batch_size
                else:
                    batch_size_curr = batch_size

                idxs = np.random.choice(idx_avail, [batch_size_curr], replace=False)
                # print(idxs)
                idxs_used_train_mode += list(idxs)
                idx_avail = [i for i in idx_avail if i not in idxs_used_train_mode]

                inputs, labels = get_batch_example(train_data, idxs, batch_size_curr, T, args.labels_train, input_size, dt, 26, True)
                inputs = inputs.permute(1, 0, 2).to(args.device)

                binary_model.init(inputs, burnin=burnin)

                readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

                for t in range(burnin, T):
                    s, r, u = binary_model(inputs[t])

                    for l, ro_h in enumerate(readout_hist):
                        readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

                preds = torch.cat((preds, readout_hist[-1].type_as(preds)))

            np.save(os.path.join(results_path, 'train_predictions_latest_mode'), preds.numpy())
            np.save(os.path.join(results_path, 'idxs_train_mode'), np.array(idxs_used_train_mode))



        ### Mean testing
        with torch.no_grad():
            n_batchs_test = n_samples_test // batch_size + (1 - (n_samples_test % batch_size == 0))
            idx_avail_test = samples_test
            idxs_used_test_mean = []

            print('Mean testing on test data epoch %d/%d' % (epoch + 1, args.n_epochs))
            predictions_mean = torch.FloatTensor()
            labels_mean = torch.FloatTensor()

            # for i in tqdm(range(n_batchs_test)):
            for i in range(n_batchs_test):
                if (i == (n_batchs_test - 1)) & (n_samples_test % batch_size != 0):
                    batch_size_curr = n_samples_test % batch_size
                else:
                    batch_size_curr = batch_size

                idxs_test = np.random.choice(idx_avail_test, [batch_size_curr], replace=False)
                idxs_used_test_mean += list(idxs_test)
                idx_avail_test = [i for i in idx_avail_test if i not in idxs_used_test_mean]

                inputs, _ = get_batch_example(test_data, idxs_test, batch_size_curr, T, args.labels_test, input_size, dt, 26, True)
                inputs = inputs.permute(1, 0, 2).to(args.device)
                predictions_batch = torch.zeros([batch_size_curr, 10, T - burnin, len(args.labels_train)])

                for j in range(10):
                    optimizer.update_concrete_weights(test=True)

                    binary_model.init(inputs, burnin=burnin)

                    readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

                    for t in range(burnin, T):
                        # forward pass: compute predicted outputs by passing inputs to the model
                        s, r, u = binary_model(inputs[t])

                        for l, ro_h in enumerate(readout_hist):
                            readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

                    predictions_batch[:, j] = readout_hist[-1].permute(1, 0, 2)
                predictions_mean = torch.cat((predictions_mean, predictions_batch))

            np.save(os.path.join(results_path, 'test_predictions_latest_mean'), predictions_mean.numpy())
            np.save(os.path.join(results_path, 'idxs_test_mean'), np.array(idxs_used_test_mean))


            n_batchs = n_samples_train // batch_size + (1 - (n_samples_train % batch_size == 0))
            idx_avail = samples_train
            idxs_used_train_mean = []
            predictions_mean = torch.FloatTensor()
            labels_mean = torch.FloatTensor()

            print('Mean testing on train data epoch %d/%d' % (epoch + 1, args.n_epochs))

            # for i in tqdm(range(n_batchs)):
            for i in range(n_batchs):
                if (i == (n_batchs - 1)) & (n_samples_train % batch_size != 0):
                    batch_size_curr = n_samples_train % batch_size
                else:
                    batch_size_curr = batch_size

                idxs = np.random.choice(idx_avail, [batch_size_curr], replace=False)
                idxs_used_train_mean += list(idxs)
                idx_avail = [i for i in idx_avail if i not in idxs_used_train_mean]

                inputs, labels = get_batch_example(train_data, idxs, batch_size_curr, T, args.labels_train, input_size, dt, 26, True)
                inputs = inputs.permute(1, 0, 2).to(args.device)
                predictions_batch = torch.zeros([batch_size_curr, 10, T - burnin, len(args.labels_train)])

                for j in range(10):
                    optimizer.update_concrete_weights(test=True)
                    binary_model.init(inputs, burnin=burnin)

                    readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

                    for t in range(burnin, T):
                        s, r, u = binary_model(inputs[t])

                        for l, ro_h in enumerate(readout_hist):
                            readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

                    predictions_batch[:, j] = readout_hist[-1].permute(1, 0, 2)
                predictions_mean = torch.cat((predictions_mean, predictions_batch))

            np.save(os.path.join(results_path, 'train_predictions_latest_mean'), predictions_mean.numpy())
            np.save(os.path.join(results_path, 'idxs_train_mean'), np.array(idxs_used_train_mean))

