import argparse
import tables
import numpy as np
from data_preprocessing.load_data import create_dataloader

from snn.utils.misc import find_indices_for_labels, str2bool
from pytorch_memlab import MemReporter

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects")
    parser.add_argument('--dataset', default=r"mnist_dvs")
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--test_period', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--polarity', type=str, default='true')


    args = parser.parse_args()

args.polarity = str2bool(args.polarity)

sample_length = 2e6  # length of samples during training in ms
dt = 1000  # us
T = int(sample_length / dt)  # number of timesteps in a sample
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
                                      sample_length_test=sample_length, dt=dt, polarity=args.polarity, num_workers=0)
train_iterator = iter(train_dl)
test_iterator = iter(test_dl)

for epoch in range(args.n_epochs):
    print('Epoch  %d' % epoch)
    inputs, labels = next(train_iterator)

    inputs = inputs.transpose(0, 1)
    labels = labels

    print(inputs.shape, labels.shape)

    if (epoch + 1) % args.test_period == 0:
        for inputs, labels in test_iterator:
            print(inputs.shape, labels.shape)

