import argparse
from copy import deepcopy
import numpy as np
import os
import torch
import yaml

from model.LIF_MLP import LIFMLP

from utils.data_utils import make_1d_signal, CustomDataset
from utils.loss import DECOLLELoss
from utils.misc import make_experiment_dir, get_optimizer
from utils.test_utils import launch_tests
from utils.train_utils import train_on_example


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    parser.add_argument('--home', default=r"\home")
    parser.add_argument('--params_file', default=r"BayesBiSNN\experiments\parameters\params_1d_bbisnn.yml")

    args = parser.parse_args()

print(args)

params_file = os.path.join(args.home, args.params_file)
with open(params_file, 'r') as f:
    params = yaml.load(f)

results_path = make_experiment_dir(args.home + '/results', '1d_experiment', params)

# Activate cuda if relevant
if not params['disable_cuda'] and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Create dataloaders
x_train, y_train, x_test, y_test, x_train_bin, x_test_bin = make_1d_signal(T=params['T'], step_train=20, step_test=100, n_neuron_per_dim=params['n_neurons_per_dim'])
train_dataset = CustomDataset(x_train_bin, y_train)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

test_dataset = CustomDataset(x_test_bin, y_test)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
n_examples_train = len(x_train)
n_examples_test = len(x_test)
input_size = [x_train_bin.shape[-1]]

# Save test/train data
np.save(os.path.join(results_path, 'x_train'), x_train.numpy())
np.save(os.path.join(results_path, 'y_train'), y_train.numpy())
np.save(os.path.join(results_path, 'x_test'), x_test.numpy())
np.save(os.path.join(results_path, 'y_test'), y_test.numpy())


# Create (relaxed) binary model and latent model
binary_model = LIFMLP(input_size,
                      1,
                      n_neurons=[128, 128],
                      with_bias=False,
                      prior_p=0.5,
                      softmax=False
                      ).to(device)
latent_model = deepcopy(binary_model)

# specify loss function
criterion = [torch.nn.SmoothL1Loss() for _ in range(binary_model.num_layers)]

decolle_loss = DECOLLELoss(criterion, latent_model)

# specify optimizer
optimizer = get_optimizer(params, binary_model, latent_model, device)

# Training
for epoch in range(params['n_epochs']):
    print('Epoch %d/%d' % (epoch + 1, params['n_epochs']))
    preds = torch.FloatTensor()
    true_labels = torch.FloatTensor()

    train_iterator = iter(train_dl)
    test_iterator = iter(test_dl)

    for inputs, labels in train_iterator:
        inputs = inputs.transpose(0, 1).to(device)
        labels = labels.unsqueeze(1).unsqueeze(2).repeat(1, 1, params['T']).to(device)

        readout_hist = train_on_example(binary_model, optimizer, decolle_loss, inputs, labels, params['burnin'], params['T'])

    torch.save(binary_model.state_dict(), results_path + '/binary_model_weights.pt')
    torch.save(latent_model.state_dict(), results_path + '/latent_model_weights.pt')


    if (epoch + 1) % (params['n_epochs'] // 5) == 0:
        torch.save(binary_model.state_dict(), results_path + '/binary_model_weights_%d.pt' % (1 + epoch))
        torch.save(latent_model.state_dict(), results_path + '/latent_model_weights_%d.pt' % (1 + epoch))


    # Testing
    if (epoch + 1) % params['test_period'] == 0:
        launch_tests(binary_model, optimizer, params['burnin'], None, test_dl, params['T'], epoch, params, device, results_path)
