import fnmatch
import math
import numpy as np
import os
import time
import torch
from torch.nn.init import _calculate_correct_fan, calculate_gain

from optimizer.BBSNN import BayesBiSNNRP
from optimizer.STBiSNN import BiSGD
from utils.binarize import binarize


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


def get_acc(preds, labels, batch_size):
    with torch.no_grad():
        acc = torch.sum(preds == torch.sum(labels.cpu(), dim=-1).argmax(dim=1)).float() / batch_size

        return acc

def make_experiment_dir(path, exp_type, params):
    prelist = np.sort(fnmatch.filter(os.listdir(path), '[0-9][0-9][0-9]__*'))
    if len(prelist) == 0:
        expDirN = "001"
    else:
        expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

    if params['optimizer_type'] == 'BayesBiSNN':
        results_path = time.strftime(path + r'/' + expDirN + "__" + "%d-%m-%Y",
                                     time.localtime()) + '_' + exp_type + r'_%d_epochs' % params['n_epochs'] \
                       + '_temp_%3f' % params['tau'] + '_prior_%3f' % params['prior_p'] + '_rho_%f' % params['rho'] + '_lr_%f' % params['lr']
    elif params['optimizer_type'] == 'STBiSNN':
        results_path = time.strftime(path + r'/' + expDirN + "__" + "%d-%m-%Y",
                                     time.localtime()) + '_' + exp_type + r'_%d_epochs' % params['n_epochs']  + '_lr_%f' % params['lr']
    os.makedirs(results_path)

    return results_path

def get_optimizer(params, binary_model, latent_model, device):
    if params['optimizer_type'] == 'BayesBiSNN':
        optimizer = BayesBiSNNRP(binary_model.parameters(), latent_model.parameters(), lr=params['lr'],
                                 tau=params['tau'], prior_p=params['prior_p'], rho=params['rho'], device=device)
    elif params['optimizer_type'] == 'STBiSNN':
        optimizer = BiSGD(binary_model.parameters(), latent_model.parameters(), lr=params['lr'], binarizer=binarize)

    return optimizer

def get_scale(layer, scaling=True):
    if scaling and hasattr(layer, 'weight'):
        fan = _calculate_correct_fan(layer.weight, mode='fan_in')
        gain = calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
        std = gain / math.sqrt(fan)
        scale = math.sqrt(3.0) * std
    else:
        scale = 1.

    return scale
