import numpy as np
import torch

def make_moon_dataset(n_samples, T, noise, n_neuron_per_dim, res=100):
    '''
    Generates points from the Two Moons dataset, rescale them in [0, 1] x [0, 1] and encodes them using population coding
    '''

    from sklearn import datasets
    data = datasets.make_moons(n_samples=n_samples, noise=noise)

    c_intervals = res / max(n_neuron_per_dim - 1, 1)
    c = np.arange(0, res + c_intervals, c_intervals)

    data[0][:, 0] = (data[0][:, 0] - np.min(data[0][:, 0])) / (np.max(data[0][:, 0]) - np.min(data[0][:, 0]))
    data[0][:, 1] = (data[0][:, 1] - np.min(data[0][:, 1])) / (np.max(data[0][:, 1]) - np.min(data[0][:, 1]))

    binary_inputs = torch.zeros([len(data[0]), T, 2 * n_neuron_per_dim])
    binary_outputs = torch.zeros([len(data[0]), T, 1])

    for i, sample in enumerate(data[0]):
        rates_0 = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[0] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, :n_neuron_per_dim] = torch.bernoulli(torch.tensor(rates_0).unsqueeze(0).repeat(T, 1))

        rates_1 = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[1] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, n_neuron_per_dim:] = torch.bernoulli(torch.tensor(rates_1).unsqueeze(0).repeat(T, 1))

        binary_outputs[i, :] = data[1][i]

    return binary_inputs, binary_outputs, torch.FloatTensor(data[0]), torch.FloatTensor(data[1])

def make_moon_test(n_samples_per_dim, T, n_neuron_per_dim, res=100):
    '''
    Generates a grid of equally spaced points in [0, 1] x [0, 1] and encodes them as binary signals using population coding
    '''

    n_samples = n_samples_per_dim ** 2

    c_intervals = res / max(n_neuron_per_dim - 1, 1)
    c = np.arange(0, res + c_intervals, c_intervals)

    binary_inputs = torch.zeros([n_samples, T, 2 * n_neuron_per_dim])

    y, x = np.meshgrid(np.arange(n_samples_per_dim), np.arange(n_samples_per_dim))
    x = (x / n_samples_per_dim).flatten()
    y = (y / n_samples_per_dim).flatten()

    for i in range(n_samples):
        rates_0 = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (x[i] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, :n_neuron_per_dim] = torch.bernoulli(torch.tensor(rates_0).unsqueeze(0).repeat(T, 1))

        rates_1 = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (y[i] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, n_neuron_per_dim:] = torch.bernoulli(torch.tensor(rates_1).unsqueeze(0).repeat(T, 1))

    return binary_inputs, torch.FloatTensor(x), torch.FloatTensor(y)


def make_1d_signal(T=100, step_train=100, step_test=100, n_neuron_per_dim=10, res=100):
    ''''
    Generates a 1D signal, rescale the points in [0, 1] and encodes them using population coding
    '''

    x0 = np.arange(-1, 0, 1 / step_train)
    x1 = np.arange(1.5, 2.5, 1 / step_train)
    x2 = np.arange(4, 5, 1 / step_train)
    x_train = np.concatenate([x0, x1, x2])

    x_test = np.arange(-1, 5, 1 / step_test)

    def function(x):
        return x - 0.1 * x ** 2 + np.cos(np.pi * x / 2)

    y_train = function(x_train)
    y_test = function(x_test)

    noise_std = 0.1
    noise_train = np.random.randn(*x_train.shape) * noise_std
    y_train = y_train + noise_train
    y_train = (y_train - np.min(y_test)) / (np.max(y_test) - np.min(y_test))
    x_train = (x_train - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

    y_test = (y_test - np.min(y_test)) / (np.max(y_test) - np.min(y_test))
    x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

    ### Population coding
    # Centers of the cosine basis
    c_intervals = 4 * (res / max(n_neuron_per_dim - 1, 1))
    c = np.arange(0, res + c_intervals, c_intervals / 4)

    x_train_bin = torch.zeros([len(x_train), T, n_neuron_per_dim])

    for i, sample in enumerate(zip(x_train, y_train)):
        rates_x = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[0] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        x_train_bin[i] = torch.bernoulli(torch.tensor(rates_x).unsqueeze(0).repeat(T, 1))

    x_test_bin = torch.zeros([len(x_test), T, n_neuron_per_dim])

    for i, sample in enumerate(zip(x_test, y_test)):
        rates_x = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[0] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        x_test_bin[i] = torch.bernoulli(torch.tensor(rates_x).unsqueeze(0).repeat(T, 1))

    return torch.FloatTensor(x_train), torch.FloatTensor(y_train), torch.FloatTensor(x_test), torch.FloatTensor(y_test), x_train_bin, x_test_bin

class CustomDataset(torch.utils.data.Dataset):
    '''
    Wrapper to create dataloaders from the synthetic datasets
    '''

    def __init__(self, data, target):
        self.data = data
        self.target = target
        super(CustomDataset, self).__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return self.data[key], self.target[key]
