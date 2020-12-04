import torch
from utils.loss import DECOLLELoss, one_hot_crossentropy
from model.LIF_MLP import LIFMLP
from tqdm import tqdm
from optimizer.BBSNN import BayesBiSNNRF
from copy import deepcopy
from utils.binarize import binarize, binarize_stochastic
import os
from torch.optim.lr_scheduler import StepLR
import argparse
import tables
import numpy as np
from data_preprocessing.load_data_old import get_batch_example
from collections import Counter

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

    parser.add_argument('--lr', type=float, default=0.01)
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
batch_size = 8
sample_length = 2000  # length of samples during training in ms
dt = 5000  # us
T = int(sample_length * 1000 / dt)  # number of timesteps in a sample
input_size = [676]
n_classes = 10
burnin = 100

dataset = tables.open_file(home + r'/datasets/mnist-dvs/mnist_dvs_events.hdf5')
train_data = dataset.root.train
test_data = dataset.root.test


binary_model = LIFMLP(input_size,
                      n_classes,
                      n_neurons=[512, 256],
                      with_output_layer=False,
                      with_bias=False,
                      scaling=True
                      ).to(args.device)

latent_model = deepcopy(binary_model)


# specify loss function
criterion = [torch.nn.SmoothL1Loss() for _ in range(binary_model.num_layers)]
if binary_model.with_output_layer:
    criterion[-1] = one_hot_crossentropy

decolle_loss = DECOLLELoss(criterion, latent_model)

# specify optimizer
optimizer = BayesBiSNNRF(binary_model.parameters(), latent_model.parameters(), lr=args.lr, device=args.device)

binary_model.init_parameters()
torch.save(binary_model.state_dict(), os.getcwd() + '/results/binary_model_weights.pt')

# print(binary_model.scales)
# print([layer.scale for layer in binary_model.LIF_layers])

for epoch in range(n_epochs):
    loss = 0

    idxs = np.random.choice(np.arange(9000), [batch_size], replace=False)

    inputs, labels = get_batch_example(train_data, idxs, batch_size, T, n_classes, input_size, dt, 26, False)

    inputs = inputs.permute(1, 0, 2).to(args.device)
    labels = labels.to(args.device)

    optimizer.update_concrete_weights()

    binary_model.init(inputs, burnin=burnin)

    readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

    # print([Counter(w.detach().numpy().flatten()) for w in binary_model.parameters()])

    print('Epoch %d/%d' % (epoch, n_epochs))
    for t in tqdm(range(burnin, T)):
        # forward pass: compute new pseudo-binary weights
        optimizer.update_concrete_weights()
        # print(list(binary_model.parameters()))

        # forward pass: compute predicted outputs by passing inputs to the model
        s, r, u = binary_model(inputs[t])

        for l, ro_h in enumerate(readout_hist):
            readout_hist[l] = torch.cat((ro_h, r[l].unsqueeze(0)), dim=0)

        # calculate the loss
        loss = decolle_loss(s, r, u, target=labels[:, :, t])
        loss.backward()
        optimizer.step(loss.detach())

        optimizer.zero_grad()

    with torch.no_grad():
        # print(u[-1])
        # print(r[-1])
        print(torch.sum(readout_hist[-1], dim=0))
        # print(Counter(list(binary_model.parameters())[-1].numpy().flatten()), Counter(list(binary_model.parameters())[-2].numpy().flatten()))
        # print(Counter(list(binary_model.parameters())[-3].numpy().flatten()), Counter(list(binary_model.parameters())[-4].numpy().flatten()))
        # print(list(binary_model.parameters()))

        print(torch.sum(readout_hist[-1], dim=0).argmax(dim=1))
        print(torch.sum(labels, dim=-1).argmax(dim=1))
        acc = torch.sum(torch.sum(readout_hist[-1], dim=0).argmax(dim=1) == torch.sum(labels, dim=-1).argmax(dim=1)).float() / batch_size
        # backward pass: compute gradient of the loss with respect to model parameters
        print(acc)

    # if (epoch + 1) % 100 == 0:
    #     torch.save(binary_model.state_dict(), os.getcwd() + '/results/binary_model_weights.pt')
    # if (epoch + 1) % 10 == 0:
    #     with torch.no_grad():
    #         n_batchs_test = 1000 // batch_size
    #         idx_avail = [i for i in range(1000)]
    #         labels_test = torch.LongTensor()
    #         predictions = torch.LongTensor()
    #
    #         for i in range(n_batchs_test):
    #             idxs_test = np.random.choice(idx_avail, [batch_size], replace=False)
    #             idx_avail = [i for i in idx_avail if i not in idxs_test]
    #
    #             inputs, labels = get_batch_example(test_data, idxs_test, batch_size, T, n_classes, input_size, dt, 26, False)
    #             inputs = inputs.permute(1, 0, 2).to(args.device)
    #             labels = labels.to(args.device)
    #
    #             binary_model.init(inputs, burnin=burnin)
    #
    #             readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]
    #
    #             print('Batch %d/%d' % (i, n_batchs_test))
    #             for t in tqdm(range(burnin, T)):
    #                 # forward pass: compute new pseudo-binary weights
    #                 optimizer.update_concrete_weights()
    #
    #                 # forward pass: compute predicted outputs by passing inputs to the model
    #                 s, r, u = binary_model(inputs[t])
    #
    #                 for l, ro_h in enumerate(readout_hist):
    #                     readout_hist[l] = torch.cat((ro_h, r[l].unsqueeze(0)), dim=0)
    #
    #             predictions = torch.cat((predictions, torch.sum(readout_hist[-1], dim=0).argmax(dim=1)))
    #             labels_test = torch.cat((labels_test, torch.sum(labels, dim=-1).argmax(dim=1)))
    #
    #         acc = torch.sum(predictions == labels_test).float() / (batch_size * n_batchs_test)
    #         print('Epoch %d/ test acc %f' % (epoch, acc))
