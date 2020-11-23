import numpy as np
import torch
import argparse

def get_output_shape(input_shape, kernel_size=[3,3], stride = [1,1], padding=[1,1], dilation=[0,0]):
    if not hasattr(kernel_size, '__len__'):
        kernel_size = [kernel_size, kernel_size]
    if not hasattr(stride, '__len__'):
        stride = [stride, stride]
    if not hasattr(padding, '__len__'):
        padding = [padding, padding]
    if not hasattr(dilation, '__len__'):
        dilation = [dilation, dilation]
    im_height = input_shape[-2]
    im_width = input_shape[-1]
    height = int((im_height + 2 * padding[0] - dilation[0] *
                  (kernel_size[0] - 1) - 1) // stride[0] + 1)
    width = int((im_width + 2 * padding[1] - dilation[1] *
                  (kernel_size[1] - 1) - 1) // stride[1] + 1)
    return [height, width]


def state_detach(state):
    for s in state:
        s.detach_()


def make_moon_dataset_bin(n_samples, T, noise):
    from sklearn import datasets

    data = datasets.make_moons(n_samples=n_samples, noise=noise)

    data[0][:, 0] = (data[0][:, 0] - np.min(data[0][:, 0])) / (np.max(data[0][:, 0]) - np.min(data[0][:, 0]))
    data[0][:, 1] = (data[0][:, 1] - np.min(data[0][:, 1])) / (np.max(data[0][:, 1]) - np.min(data[0][:, 1]))

    binary_inputs = torch.zeros([len(data[0]), T, 2])
    binary_outputs = torch.zeros([len(data[0]), T, 2])

    for i, sample in enumerate(data[0]):
        binary_inputs[i, :, 0] = torch.bernoulli(torch.tensor([sample[0]] * T))
        binary_inputs[i, :, 1] = torch.bernoulli(torch.tensor([sample[1]] * T))

        binary_outputs[i, :, data[1][i]] = 1

    return binary_inputs, binary_outputs


def make_moon_dataset_bin_pop_coding(n_samples, T, noise, n_neuron_per_dim, res=100):
    from sklearn import datasets
    data = datasets.make_moons(n_samples=n_samples, noise=noise)

    c_intervals = res / max(n_neuron_per_dim - 1, 1)
    c = np.arange(0, res + c_intervals, c_intervals)

    data[0][:, 0] = (data[0][:, 0] - np.min(data[0][:, 0])) / (np.max(data[0][:, 0]) - np.min(data[0][:, 0]))
    data[0][:, 1] = (data[0][:, 1] - np.min(data[0][:, 1])) / (np.max(data[0][:, 1]) - np.min(data[0][:, 1]))

    binary_inputs = torch.zeros([len(data[0]), T, 2 * n_neuron_per_dim])
    binary_outputs = torch.zeros([len(data[0]), T, 2])

    for i, sample in enumerate(data[0]):
        rates_0 = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[0] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, :n_neuron_per_dim] = torch.bernoulli(torch.tensor(rates_0).unsqueeze(0).repeat(T, 1))

        rates_1 = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[1] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        binary_inputs[i, :, n_neuron_per_dim:] = torch.bernoulli(torch.tensor(rates_1).unsqueeze(0).repeat(T, 1))

        binary_outputs[i, :, data[1][i]] = 1

    return binary_inputs, binary_outputs, data[0], data[1]


def make_moon_test_dataset_bin_pop_coding(n_samples_per_dim, T, n_neuron_per_dim, res=100):
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

    return binary_inputs, x, y


def gen_1d_signal(T=100, step=100, n_neuron_per_dim=10, res=100):
    x0 = np.arange(-1, 0, 1 / step)
    x1 = np.arange(1.5, 2.5, 1 / step)
    x2 = np.arange(4, 5, 1 / step)
    x_train = np.concatenate([x0, x1, x2])

    x_test = np.arange(-1, 5, 1 / step)

    def function(x):
        return x - 0.1 * x ** 2 + np.cos(np.pi * x / 2)

    y_train = function(x_train)
    y_test = function(x_test)

    noise_std = 0.25
    noise_train = np.random.randn(*x_train.shape) * noise_std
    y_train = y_train + noise_train
    y_train = (y_train - np.min(y_train)) / (np.max(y_train) - np.min(y_train))
    x_train = (x_train - np.min(x_train)) / (np.max(x_train) - np.min(x_train))

    y_test = (y_test - np.min(y_test)) / (np.max(y_test) - np.min(y_test))
    x_test = (x_test - np.min(x_test)) / (np.max(x_test) - np.min(x_test))

    ### Population coding
    # Centers of the cosine basis
    c_intervals = 4 * (res / max(n_neuron_per_dim - 1, 1))
    c = np.arange(0, res + c_intervals, c_intervals / 4)

    x_train_bin = torch.zeros([len(x_train), T, n_neuron_per_dim])
    y_train_bin = torch.zeros([len(y_train), T, n_neuron_per_dim])

    for i, sample in enumerate(zip(x_train, y_train)):
        rates_x = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[0] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        x_train_bin[i] = torch.bernoulli(torch.tensor(rates_x).unsqueeze(0).repeat(T, 1))

        rates_y = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[1] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        y_train_bin[i] = torch.bernoulli(torch.tensor(rates_y).unsqueeze(0).repeat(T, 1))

    x_test_bin = torch.zeros([len(x_test), T, n_neuron_per_dim])
    y_test_bin = torch.zeros([len(y_test), T, n_neuron_per_dim])

    for i, sample in enumerate(zip(x_test, y_test)):
        rates_x = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[0] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        x_test_bin[i] = torch.bernoulli(torch.tensor(rates_x).unsqueeze(0).repeat(T, 1))

        rates_y = np.array([0.5 + np.cos(max(-np.pi, min(np.pi, np.pi * (sample[1] * res - c[k]) / c_intervals))) / 2 for k in range(n_neuron_per_dim)]).T
        y_test_bin[i] = torch.bernoulli(torch.tensor(rates_y).unsqueeze(0).repeat(T, 1))

    return x_train, y_train, x_test, y_test, x_train_bin, y_train_bin, x_test_bin, y_test_bin


def gen_1d_signal_realtarget(T=100, step=100, n_neuron_per_dim=10, res=100):
    x0 = np.arange(-1, 0, 1 / step)
    x1 = np.arange(1.5, 2.5, 1 / step)
    x2 = np.arange(4, 5, 1 / step)
    x_train = np.concatenate([x0, x1, x2])

    x_test = np.arange(-1, 5, 1 / step)

    def function(x):
        return x - 0.1 * x ** 2 + np.cos(np.pi * x / 2)

    y_train = function(x_train)
    y_test = function(x_test)

    noise_std = 0.25
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

    return x_train, y_train, x_test, y_test, x_train_bin, x_test_bin



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_acc(preds, labels, batch_size):
    with torch.no_grad():
        acc = torch.sum(preds == torch.sum(labels.cpu(), dim=-1).argmax(dim=1)).float() / batch_size

        return acc

