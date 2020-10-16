import torch
from utils.loss import DECOLLELoss
from model.LeNet import LenetLIF
from tqdm import tqdm
from optimizer.BiSGD import BiSGD
from copy import deepcopy
from utils.binarize import binarize, binarize_stochastic
import os
from torch.optim.lr_scheduler import StepLR
import argparse
import tables
import numpy as np
from data_preprocessing.load_data import get_batch_example

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
    parser.add_argument('--where', default='local')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--disable-cuda', type=str, default='true', help='Disable CUDA')

    args = parser.parse_args()


if args.where == 'local':
    home = r'C:/Users/K1804053/PycharmProjects'
elif args.where == 'rosalind':
    home = r'/users/k1804053'
elif args.where == 'jade':
    home = r'/jmain01/home/JAD014/mxm09/nxs94-mxm09'
elif args.where == 'gcloud':
    home = r'/home/k1804053'


args.disable_cuda = str2bool(args.disable_cuda)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')


n_epochs = 5000
test_period = 1
batch_size = 32
sample_length = 2000  # length of samples during training in ms
dt = 5000  # us
T = int(sample_length * 1000 / dt)  # number of timesteps in a sample
input_size = [2, 26, 26]
n_classes = 10

dataset = tables.open_file(home + r'/datasets/mnist-dvs/mnist_dvs_events.hdf5')
train_data = dataset.root.train
test_data = dataset.root.test

binary_model = LenetLIF(input_size,
                        Nhid_conv=[64, 128, 128],
                        Nhid_mlp=[0],
                        out_channels=10,
                        kernel_size=[7],
                        stride=[1],
                        pool_size=[2, 1, 2],
                        dropout=[0.5],
                        num_conv_layers=3,
                        num_mlp_layers=0
                        ).to(args.device)

latent_model = deepcopy(binary_model).to(args.device)


# specify loss function
criterion = [torch.nn.SmoothL1Loss()]
decolle_loss = DECOLLELoss(criterion, latent_model)

# specify optimizer
optimizer = BiSGD(binary_model.parameters(), latent_model.parameters(), lr=args.lr, binarizer=binarize)
scheduler = StepLR(optimizer, step_size=500, gamma=0.5)

binary_model.init_parameters()
torch.save(binary_model.state_dict(), os.getcwd() + '/results/binary_model_weights.pt')

for epoch in range(n_epochs):
    loss = 0

    idxs = np.random.choice(np.arange(9000), [batch_size], replace=False)

    inputs, labels = get_batch_example(train_data, idxs, batch_size, T, n_classes, input_size, dt, 26, False)

    inputs = inputs.permute(1, 0, 2, 3, 4).to(args.device)
    labels = labels.to(args.device)

    binary_model.init(inputs, burnin=100)

    readout_hist = [torch.Tensor().to(args.device) for _ in range(len(binary_model.readout_layers))]

    print('Epoch %d/%d' % (epoch, n_epochs))
    for t in tqdm(range(100, T)):
        # forward pass: compute predicted outputs by passing inputs to the model
        s, r, u = binary_model(inputs[t])

        for l, ro_h in enumerate(readout_hist):
            readout_hist[l] = torch.cat((ro_h, r[l].unsqueeze(0)), dim=0)

        # calculate the loss
        loss = decolle_loss(s, r, u, target=labels[:, :, t])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    print(torch.sum(readout_hist[-1], dim=0).argmax(dim=1),  torch.sum(labels, dim=-1).argmax(dim=1))
    acc = torch.sum(torch.sum(readout_hist[-1], dim=0).argmax(dim=1) == torch.sum(labels, dim=-1).argmax(dim=1)).float() / batch_size
    # backward pass: compute gradient of the loss with respect to model parameters
    print(acc)
