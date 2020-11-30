import torch
from utils.loss import DECOLLELoss, one_hot_crossentropy
from model.LeNet import LenetLIF
from tqdm import tqdm
from optimizer.BiSGD import BiSGD
from copy import deepcopy
from utils.binarize import binarize
import os
from torch.optim.lr_scheduler import StepLR
import argparse
import tables
import numpy as np
from data_preprocessing.load_data import get_batch_example
from utils.activations import smooth_sigmoid, smooth_step
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
    parser.add_argument('--n_epochs', type=int, default=20000)
    parser.add_argument('--test_period', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_samples', type=int, default=10)

    parser.add_argument('--lr', type=float, default=100)
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
                             time.localtime()) + '_' + 'mnist_dvs_stlr_lenet_' + r'_%d_epochs' % args.n_epochs + '_lr_%f' % args.lr
os.makedirs(results_path)

args.with_softmax = str2bool(args.with_softmax)
args.disable_cuda = str2bool(args.disable_cuda)


args.disable_cuda = str2bool(args.disable_cuda)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

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

# if binary_model.with_output_layer:
#     criterion[-1] = one_hot_crossentropy

decolle_loss = DECOLLELoss(criterion, latent_model)

# specify optimizer
optimizer = BiSGD(binary_model.parameters(), latent_model.parameters(), lr=args.lr, binarizer=binarize)

binary_model.init_parameters()
optimizer.step()  # binarize weights


for epoch in range(args.n_epochs):
    binary_model.softmax = args.with_softmax
    torch.save(binary_model.state_dict(), os.getcwd() + '/results/binary_model_weights.pt')

    loss = 0

    idxs = np.random.choice(np.arange(n_examples_train), [args.batch_size], replace=False)

    inputs, labels = get_batch_example(train_data, idxs, args.batch_size, T, args.labels, input_size, dt, 26, args.polarity)

    inputs = inputs.transpose(0, 1).to(args.device)
    labels = labels.to(args.device)

    binary_model.init(inputs, burnin=burnin)

    readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]


    print('Epoch %d/%d' % (epoch, args.n_epochs))
    for t in tqdm(range(burnin, T)):
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
        acc = torch.sum(torch.sum(readout_hist[-1], dim=0).argmax(dim=1) == torch.sum(labels.cpu(), dim=-1).argmax(dim=1)).float() / args.batch_size
        # backward pass: compute gradient of the loss with respect to model parameters
        print(acc)


    if (epoch + 1) % (args.n_epochs//5) == 0:
        torch.save(binary_model.state_dict(), results_path + '/binary_model_weights_%d.pt' % (1 + epoch))
        torch.save(latent_model.state_dict(), results_path + '/latent_model_weights_%d.pt' % (1 + epoch))

    if (epoch + 1) % args.test_period == 0:
        binary_model.softmax = False

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

                inputs, _ = get_batch_example(test_data, idxs, batch_size_curr, T, labels, input_size, dt, 26, args.polarity)
                inputs = inputs.transpose(0, 1).to(args.device)

                binary_model.init(inputs, burnin=burnin)

                readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

                for t in range(burnin, T):
                    # forward pass: compute predicted outputs by passing inputs to the model
                    s, r, u = binary_model(inputs[t])

                    for l, ro_h in enumerate(readout_hist):
                        readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

                predictions = torch.cat((predictions, readout_hist[output].transpose(0, 1)))

            np.save(os.path.join(results_path, 'test_predictions_latest'), predictions.numpy())
            np.save(os.path.join(results_path, 'idxs_test'), np.array(idxs_used_test))

            n_batchs_train = n_examples_train // args.batch_size + (1 - (n_examples_train % args.batch_size == 0))
            idx_avail_train = np.arange(n_examples_train)
            idxs_used_train = []
