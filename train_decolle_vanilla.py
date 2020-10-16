import torch
from utils.loss import DECOLLELoss
from model.LIF_MLP import LIFMLP
from tqdm import tqdm
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
    parser.add_argument('--lr', type=float, default=1e-3)
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
input_size = [676]
n_classes = 10

dataset = tables.open_file(home + r'/datasets/mnist-dvs/mnist_dvs_events.hdf5')
train_data = dataset.root.train
test_data = dataset.root.test


model = LIFMLP(input_shape=input_size,
               output_shape=10,
               n_neurons=[512, 256],
               num_layers=2)


# specify loss function

# specify optimizer (stochastic gradient descent) and learning rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
optimizer = torch.optim.Adamax(model.get_trainable_parameters(), lr=1e-8, betas=[0., .95])

criterion = [torch.nn.SmoothL1Loss()]
decolle_loss = DECOLLELoss(criterion, model)

model.init_parameters()

for epoch in range(n_epochs):
    loss = 0

    idxs = np.random.choice(np.arange(9000), [batch_size], replace=False)
    inputs, labels = get_batch_example(train_data, idxs, batch_size, T, n_classes, input_size, dt, 26, False)

    print(inputs.shape)
    inputs = inputs.permute(1, 0, 2).to(args.device)
    labels = labels.to(args.device)

    model.init(inputs, burnin=100)

    readout_hist = [torch.Tensor() for _ in range(len(model.readout_layers))]

    print('Epoch %d/%d' % (epoch, n_epochs))
    for t in tqdm(range(T)):
        # forward pass: compute predicted outputs by passing inputs to the model
        s, r, u = model(inputs[t])

        for l, ro_h in enumerate(readout_hist):
            readout_hist[l] = torch.cat((ro_h, r[l].unsqueeze(0)), dim=0)

        # calculate the loss
        loss = decolle_loss(s, r, u, target=labels[:, :, t])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(u[-1])
    acc = torch.sum(torch.sum(readout_hist[-1], dim=0).argmax(dim=1) == torch.sum(labels, dim=-1).argmax(dim=1)).float() / batch_size
    # backward pass: compute gradient of the loss with respect to model parameters
    print(acc)

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
