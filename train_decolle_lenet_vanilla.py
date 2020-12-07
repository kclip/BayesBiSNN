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
from utils.misc import str2bool, get_acc
from data_preprocessing.load_data import create_dataloader
from data_preprocessing.load_data_old import get_batch_example

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects")
    parser.add_argument('--results', default=r"C:\Users\K1804053\results")
    parser.add_argument('--dataset', default=r"mnist_dvs")
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--lr', type=float, default=0.1)
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
                             time.localtime()) + '_' + args.dataset + '_lenet_vanilla_' + r'_%d_epochs' % args.n_epochs + '_lr_%f' % args.lr
os.makedirs(results_path)

args.polarity = str2bool(args.polarity)
args.with_softmax = str2bool(args.with_softmax)
args.disable_cuda = str2bool(args.disable_cuda)

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

sample_length = 2e6  # length of samples during training in mus
dt = 5000  # us
T = int(sample_length / dt)
burnin = 50


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


model = LenetLIF(input_size,
                 Nhid_conv=[64, 128, 128],
                 Nhid_mlp=[],
                 out_channels=len(args.classes),
                 kernel_size=[7],
                 stride=[1],
                 pool_size=[2, 1, 2],
                 dropout=[0.],
                 num_conv_layers=3,
                 num_mlp_layers=0,
                 with_bias=True,
                 with_output_layer=False,
                 softmax=args.with_softmax).to(args.device)


# specify loss function
criterion = [one_hot_crossentropy for _ in range(model.num_layers)]
if model.with_output_layer:
    criterion[-1] = one_hot_crossentropy

decolle_loss = DECOLLELoss(criterion, model)


# specify optimizer
optimizer = torch.optim.SGD(model.get_trainable_parameters(), lr=args.lr)

model.init_parameters()

print(model.scales)
print([layer.scale for layer in model.LIF_layers])

for epoch in range(args.n_epochs):
    train_iterator = iter(train_dl)
    test_iterator = iter(test_dl)

    loss = 0

    for inputs, labels in train_iterator:
        model.softmax = args.with_softmax

        inputs = inputs.transpose(0, 1).to(args.device)
        labels = labels.to(args.device)

        model.init(inputs, burnin=burnin)

        readout_hist = [torch.Tensor() for _ in range(len(model.readout_layers))]


        print('Epoch %d/%d' % (epoch, args.n_epochs))
        for t in range(burnin, T):
            # forward pass: compute predicted outputs by passing inputs to the model
            s, r, u = model(inputs[t])

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
        torch.save(model.state_dict(), results_path + '/binary_model_weights_%d.pt' % (1 + epoch))
        torch.save(model.state_dict(), results_path + '/latent_model_weights_%d.pt' % (1 + epoch))


    if (epoch + 1) % args.test_period == 0:
        model.softmax = False

        with torch.no_grad():
            print('Testing epoch %d/%d' % (epoch + 1, args.n_epochs))
            predictions = torch.FloatTensor()
            true_labels = torch.FloatTensor()

            for inputs, labels in test_iterator:
                inputs = inputs.transpose(0, 1).to(args.device)

                model.init(inputs, burnin=burnin)

                readout_hist = [torch.Tensor() for _ in range(len(model.readout_layers))]

                for t in range(burnin, T):
                    # forward pass: compute predicted outputs by passing inputs to the model
                    s, r, u = model(inputs[t])

                    for l, ro_h in enumerate(readout_hist):
                        readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

                predictions = torch.cat((predictions, readout_hist[-1].transpose(0, 1)))
                true_labels = torch.cat((true_labels, torch.sum(labels.cpu(), dim=-1).argmax(dim=1).type_as(true_labels)))

            np.save(os.path.join(results_path, 'test_predictions_latest'), predictions.numpy())
            np.save(os.path.join(results_path, 'true_labels_test'), true_labels.numpy())
