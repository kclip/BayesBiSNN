import torch
from utils.loss import decolle_loss
from model.LIF_MLP import LIFMLP
from data_loaders.load_mnistdvs_flat import create_data
from tqdm import tqdm
from optimizer.BBSNN import BayesBiSNN
from copy import deepcopy
import os
from torch.optim.lr_scheduler import StepLR
import argparse

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser(description='Train probabilistic multivalued SNNs using Pytorch')

    # Training arguments
    parser.add_argument('--where', default='local')
    parser.add_argument('--temperate', type=float, default=0.1)

    args = parser.parse_args()


if args.where == 'local':
    home = r'C:/Users/K1804053/PycharmProjects'
elif args.where == 'rosalind':
    home = r'/users/k1804053'
elif args.where == 'jade':
    home = r'/jmain01/home/JAD014/mxm09/nxs94-mxm09'
elif args.where == 'gcloud':
    home = r'/home/k1804053'


n_epochs = 5000
batch_size = 64
sample_length = 2000  # length of samples during training in ms
dt = 5000  # us
T = int(sample_length * 1000 / dt)  # number of timesteps in a sample
input_size = 1352

gen_train, gen_test = create_data(path_to_hdf5=home + r'/datasets/mnist-dvs/mnist_dvs_events.hdf5',
                                  path_to_data=home + r'/datasets/mnist-dvs/processed_polarity',
                                  batch_size=batch_size,
                                  chunk_size=sample_length,
                                  n_inputs=1352,
                                  dt=dt)


binary_model = LIFMLP(input_shape=[1352],
                      output_shape=10,
                      n_neurons=[512, 256],
                      num_layers=2)
latent_model = deepcopy(binary_model)


# specify loss function
criterion = torch.nn.SmoothL1Loss()

# specify optimizer
optimizer = BayesBiSNN(binary_model.parameters(), latent_model.parameters(), lr=0.1, temperature=args.temperature)

binary_model.init_parameters()
torch.save(binary_model.state_dict(), os.getcwd() + '/results/binary_model_weights.pt')


for epoch in range(n_epochs):
    loss = 0

    inputs, labels = gen_train.next()
    inputs = torch.Tensor(inputs)
    labels = torch.Tensor(labels)

    binary_model.init(inputs, burnin=100)

    readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]


    print('Epoch %d/%d' % (epoch, n_epochs))
    for t in tqdm(range(T)):
        # forward pass: compute predicted outputs by passing inputs to the model
        _, r, _ = binary_model(inputs[t])

        for l, ro_h in enumerate(readout_hist):
            readout_hist[l] = torch.cat((ro_h, r[l].unsqueeze(0)), dim=0)

        # calculate the loss
        loss = torch.mean(decolle_loss(r, labels[t], criterion))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    acc = torch.sum(torch.sum(readout_hist[-1], dim=0).argmax(dim=1) == torch.sum(labels, dim=0).argmax(dim=1)).float() / batch_size
    # backward pass: compute gradient of the loss with respect to model parameters
    print(acc)

    if (epoch + 1) % 100 == 0:
        torch.save(binary_model.state_dict(), os.getcwd() + '/results/binary_model_weights.pt')

#     if (i + 1) % 100 == 0:
#         # print(list(model.parameters())[0])
#         # print(list(latent_model.parameters())[0])
#
#         acc = 0
#         for j, sample_test in enumerate(test_data):
#             output = model(sample_test)
#             pred = torch.argmax(output)
# #
# #             print(j, pred, test_label[j])
#             if pred.data.numpy() == test_label[j].numpy():
#                 acc += 1
#         acc /= len(test_data)
#         print(i, acc)
