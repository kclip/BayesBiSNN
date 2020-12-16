import argparse
from copy import deepcopy
import numpy as np
import os
import torch
import yaml

from model.LIF_MLP import LIFMLP

from utils.data_utils import make_moon_dataset, make_moon_test, CustomDataset
from utils.loss import DECOLLELoss
from utils.misc import make_experiment_dir, get_optimizer
from utils.test_utils import launch_tests
from utils.train_utils import train_on_example


if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"C:\Users\K1804053\OneDrive - King's College London\PycharmProjects")
    parser.add_argument('--params_file', default=r"BayesBiSNN\experiments\parameters\params_twomoons_stbisnn.yml")
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--weights', type=str, default=None, help='Path to existing weights (relative to home)')

    # Arguments common to all models

    args = parser.parse_args()

print(args)

params_file = os.path.join(args.home, args.params_file)
with open(params_file, 'r') as f:
    params = yaml.load(f)

results_path = make_experiment_dir(args.home + r'/results', 'twomoons_bbsnn', params)

# Activate cuda if relevant
if not params['disable_cuda'] and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


x_bin_train, y_bin_train, x_train, y_train = make_moon_dataset(params['n_examples_train'], params['T'], 0.1, params['n_neurons_per_dim'])
train_dataset = CustomDataset(x_bin_train, y_bin_train)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
np.save(os.path.join(results_path, 'x_train'), x_train.numpy())
np.save(os.path.join(results_path, 'y_train'), y_train.numpy())

x_bin_test, x_test, y_test = make_moon_test(params['n_examples_per_dim_test'], params['T'], params['n_neurons_per_dim'])
test_dataset = CustomDataset(x_bin_test, y_test)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
np.save(os.path.join(results_path, 'x_test'), x_test.numpy())
np.save(os.path.join(results_path, 'y_test'), y_test.numpy())

input_size = [x_bin_train.shape[-1]]


# Create (relaxed) binary model and latent model
binary_model = LIFMLP(input_size,
                      1,
                      n_neurons=[256, 256],
                      with_bias=False,
                      prior_p=params['prior_p'],
                      softmax=False
                      ).to(device)

latent_model = deepcopy(binary_model)

# specify loss function
criterion = [torch.nn.BCEWithLogitsLoss() for _ in range(binary_model.num_layers)]

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
        labels = labels.transpose(1, 2).to(device)

        readout_hist = train_on_example(binary_model, optimizer, decolle_loss, inputs, labels, params['burnin'], params['T'])

        with torch.no_grad():
            preds = torch.cat((preds, torch.mean(readout_hist[-1].type_as(preds), dim=(0, -1))))
            true_labels = torch.cat((true_labels, labels.cpu().type_as(true_labels)))

    # Monitor how the accuracy evolves during training
    preds = torch.sigmoid(preds)
    preds[preds > 0.5] = 1.
    preds[preds <= 0.5] = 0.
    print(torch.sum(preds == torch.mean(true_labels.cpu(), dim=(-1, -2))).float() / len(preds))


    # Save current weights if needed to resume training
    torch.save(binary_model.state_dict(), results_path + '/binary_model_weights.pt')
    torch.save(latent_model.state_dict(), results_path + '/latent_model_weights.pt')

    # Save weights periodically to plot the evolution of the distribution
    if (epoch + 1) % (params['n_epochs'] // 5) == 0:
        torch.save(binary_model.state_dict(), results_path + '/binary_model_weights_%d.pt' % (1 + epoch))
        torch.save(latent_model.state_dict(), results_path + '/latent_model_weights_%d.pt' % (1 + epoch))

    # Testing
    if (epoch + 1) % params['test_period'] == 0:
        launch_tests(binary_model, optimizer, params['burnin'], None, test_dl, params['T'], epoch, params, device, results_path)
