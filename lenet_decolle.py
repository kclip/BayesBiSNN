import argparse
from copy import deepcopy
import os
import tables
import torch
import yaml

from model.LeNet import LenetLIF

from neurodata.load_data import create_dataloader
from utils.loss import DECOLLELoss, one_hot_crossentropy
from utils.misc import make_experiment_dir, get_optimizer, get_acc
from utils.test_utils import launch_tests
from utils.train_utils import train_on_example



if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--home', default=r"C:/Users/K1804053/OneDrive - King's College London/PycharmProjects")
    parser.add_argument('--params_file', default=r"BayesBiSNN\experiments\parameters\params_decolle_mnistdvs_bbisnn.yml")
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--weights', type=str, default=None, help='Path to existing weights (relative to home)')

    # Arguments common to all models

    args = parser.parse_args()

print(args)

params_file = os.path.join(args.home, args.params_file)
with open(params_file, 'r') as f:
    params = yaml.load(f)

results_path = make_experiment_dir(args.home + '/results', 'twomoons_bbsnn', params)


# Activate cuda if relevant
if not params['disable_cuda'] and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


dataset_path = args.home + params['dataset']
dataset = tables.open_file(dataset_path)
x_max = dataset.root.stats.train_data[1] // params['ds']
input_size = [2, x_max, x_max]
dataset.close()
T_train = int(params['sample_length_train'] / params['dt'])
T_test = int(params['sample_length_test'] / params['dt'])

# Create dataloaders
train_dl, test_dl = create_dataloader(dataset_path, batch_size=params['batch_size'], size=input_size, classes=params['classes'], sample_length_train=params['sample_length_train'],
                                      sample_length_test=params['sample_length_test'], dt=params['dt'], polarity=params['polarity'], ds=params['ds'], num_workers=0)

# Create (relaxed) binary model and latent model
binary_model = LenetLIF(input_size,
                        Nhid_conv=[64, 128, 128],
                        Nhid_mlp=[],
                        out_channels=len(params['classes']),
                        kernel_size=[7],
                        stride=[1],
                        pool_size=[2, 1, 2],
                        dropout=[0.],
                        num_conv_layers=3,
                        num_mlp_layers=0,
                        with_bias=True,
                        softmax=params['with_softmax']).to(device)
latent_model = deepcopy(binary_model)

# specify loss function
criterion = [one_hot_crossentropy for _ in range(binary_model.num_layers)]

decolle_loss = DECOLLELoss(criterion, latent_model)

# specify optimizer
optimizer = get_optimizer(params, binary_model, latent_model, device)

# Training
for epoch in range(params['n_epochs']):
    train_iterator = iter(train_dl)
    test_iterator = iter(test_dl)

    print('Epoch %d/%d' % (epoch, params['n_epochs']))

    for inputs, labels in train_iterator:
        binary_model.softmax = params['with_softmax']
        loss = 0

        inputs = inputs.transpose(0, 1).to(device)
        labels = labels.to(device)

        readout_hist = train_on_example(binary_model, optimizer, decolle_loss, inputs, labels, params['burnin'], T_train)

        # Monitor how the batch accuracy evolves during training
        acc = get_acc(torch.sum(readout_hist[-1], dim=0).argmax(dim=1), labels, params['batch_size'])
        print(acc)

        # Save current weights if needed to resume training
        torch.save(binary_model.state_dict(), results_path + '/binary_model_weights.pt')
        torch.save(latent_model.state_dict(), results_path + '/latent_model_weights.pt')


    # Save weights periodically to plot the evolution of the distribution
    if (epoch + 1) % (params['n_epochs']//5) == 0:
        torch.save(binary_model.state_dict(), results_path + '/binary_model_weights_%d.pt' % (1 + epoch))
        torch.save(latent_model.state_dict(), results_path + '/latent_model_weights_%d.pt' % (1 + epoch))

    # Testing
    if (epoch + 1) % params['test_period'] == 0:
        binary_model.softmax = False
        launch_tests(binary_model, optimizer, params['burnin'], None, test_dl, T_test, epoch, params, device, results_path)
