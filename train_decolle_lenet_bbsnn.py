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
from data_preprocessing.load_data import get_batch_example
from collections import Counter
import pickle
import fnmatch
import time

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
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--rho', type=float, default=5e-7)
    parser.add_argument('--prior_p', type=float, default=0.5)
    parser.add_argument('--disable-cuda', type=str, default='false', help='Disable CUDA')

    args = parser.parse_args()

prelist = np.sort(fnmatch.filter(os.listdir(args.results), '[0-9][0-9][0-9]__*'))
if len(prelist) == 0:
    expDirN = "001"
else:
    expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

results_path = time.strftime(args.results + r'/' + expDirN + "__" + "%d-%m-%Y",
                             time.localtime()) + '_' + 'mnist_dvs_bbsnnrp_lenet_' + r'_%d_epochs' % args.n_epochs\
               + '_temp_%3f' % args.temperature + '_prior_%3f' % args.prior_p + '_rho_%f' % args.rho + '_lr_%f' % args.lr
os.makedirs(results_path)

args.disable_cuda = str2bool(args.disable_cuda)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.train_accs = {i: [] for i in range(0, args.n_epochs, 100)}
args.train_accs[args.n_epochs] = []

test_period = 1
batch_size = 32
sample_length = 2000  # length of samples during training in ms
dt = 5000  # us
T = int(sample_length * 1000 / dt)  # number of timesteps in a sample
input_size = [2, 28, 28]
n_classes = 10
burnin = 100

dataset = tables.open_file(args.home + r'/datasets/mnist-dvs/mnist_dvs_events.hdf5')
train_data = dataset.root.train
test_data = dataset.root.test


binary_model = LenetLIF(input_size,
                        Nhid_conv=[64, 128, 128],
                        Nhid_mlp=[],
                        out_channels=10,
                        kernel_size=[7],
                        stride=[1],
                        pool_size=[2, 1, 2],
                        dropout=[0.5],
                        num_conv_layers=3,
                        num_mlp_layers=0,
                        with_bias=False).to(args.device)

latent_model = deepcopy(binary_model)


# specify loss function
criterion = [torch.nn.SmoothL1Loss() for _ in range(binary_model.num_layers)]
decolle_loss = DECOLLELoss(criterion, latent_model)

# specify optimizer
optimizer = BayesBiSNNRP(binary_model.parameters(), latent_model.parameters(), lr=args.lr, temperature=args.temperature, prior_p=args.prior_p, rho=args.rho, device=args.device)

binary_model.init_parameters()

# print(binary_model.scales)
# print([layer.scale for layer in binary_model.LIF_layers])

for epoch in range(args.n_epochs):
    loss = 0

    idxs = np.random.choice(np.arange(9000), [batch_size], replace=False)

    inputs, labels = get_batch_example(train_data, idxs, batch_size, T, n_classes, input_size, dt, 26, True)

    inputs = inputs.permute(1, 0, 2, 3, 4).to(args.device)
    labels = labels.to(args.device)

    optimizer.update_concrete_weights()

    # print([Counter(w.detach().numpy().flatten()) for w in binary_model.parameters()])
    binary_model.init(inputs, burnin=burnin)

    readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]


    print('Epoch %d/%d' % (epoch, args.n_epochs))
    for t in tqdm(range(burnin, T)):
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

    with torch.no_grad():
        # print(u)
        # print(torch.sum(labels.cpu(), dim=-1).argmax(dim=1))

        print(torch.sum(readout_hist[-1], dim=0).argmax(dim=1))
        acc = torch.sum(torch.sum(readout_hist[-1], dim=0).argmax(dim=1) == torch.sum(labels.cpu(), dim=-1).argmax(dim=1)).float() / batch_size
        # backward pass: compute gradient of the loss with respect to model parameters
        print(acc)

    if (epoch + 1) % 100 == 0:
        torch.save(binary_model.state_dict(), results_path + '/binary_model_weights.pt')
        with torch.no_grad():
            n_batchs_test = 1000 // batch_size
            idx_avail = [i for i in range(1000)]
            labels_test = torch.LongTensor()
            predictions = torch.LongTensor()

            for i in range(n_batchs_test):
                idxs_test = np.random.choice(idx_avail, [batch_size], replace=False)
                idx_avail = [i for i in idx_avail if i not in idxs_test]

                inputs, labels = get_batch_example(test_data, idxs_test, batch_size, T, n_classes, input_size, dt, 26, False)
                inputs = inputs.permute(1, 0, 2, 3, 4).to(args.device)
                labels = labels.to(args.device)

                binary_model.init(inputs, burnin=burnin)

                readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

                print('Batch %d/%d' % (i, n_batchs_test))
                for t in tqdm(range(burnin, T)):
                    # forward pass: compute new pseudo-binary weights
                    optimizer.update_concrete_weights()

                    # forward pass: compute predicted outputs by passing inputs to the model
                    s, r, u = binary_model(inputs[t])

                    for l, ro_h in enumerate(readout_hist):
                        readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

                predictions = torch.cat((predictions, torch.sum(readout_hist[-1], dim=0).argmax(dim=1)))
                labels_test = torch.cat((labels_test, torch.sum(labels.cpu(), dim=-1).argmax(dim=1)))

            acc = torch.sum(predictions == labels_test).float() / (batch_size * n_batchs_test)
            args.train_accs[epoch + 1] = acc.numpy()

            with open(results_path + '/test_accs.pkl', 'wb') as f:
                pickle.dump(args.train_accs, f, pickle.HIGHEST_PROTOCOL)

            print('Epoch %d/ test acc %f' % (epoch, acc))


