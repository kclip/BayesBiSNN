import torch
from utils.loss import DECOLLELoss, one_hot_crossentropy
from model.LIF_MLP import LIFMLP
from tqdm import tqdm
import argparse
import tables
import numpy as np
from data_preprocessing.load_data import get_batch_example
from utils.misc import make_moon_dataset_bin, make_moon_dataset_bin_pop_coding
import fnmatch
import time
import os

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
    parser.add_argument('--home', default='/home')
    parser.add_argument('--save_path', type=str, default=None, help='Path to where weights are stored (relative to home)')
    parser.add_argument('--n_epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=5e-1)
    parser.add_argument('--temperature', type=float, default=5e-1)
    parser.add_argument('--disable-cuda', type=str, default='false', help='Disable CUDA')

    args = parser.parse_args()


pre = args.home + r'/results/'
prelist = np.sort(fnmatch.filter(os.listdir(pre), '[0-9][0-9][0-9]__*'))
if len(prelist) == 0:
    expDirN = "001"
else:
    expDirN = "%03d" % (int((prelist[len(prelist) - 1].split("__"))[0]) + 1)

results_path = time.strftime(pre + expDirN + "__" + "%d-%m-%Y", time.localtime()) + '_' + 'mnist_dvs_bbsnnrp' + r'_%d_epochs' % args.n_epochs
os.makedirs(results_path)

args.disable_cuda = str2bool(args.disable_cuda)
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

args.train_accs = {i: [] for i in range(0, args.n_epochs, 100)}
args.train_accs[args.n_epochs] = []


n_samples_train = 200
n_samples_test = 5000
T = 100
n_neurons_per_dim = 10

# train_inputs, train_outputs = make_moon_dataset_bin(n_samples_train, T, 0.)
# test_inputs, test_outputs = make_moon_dataset_bin(n_samples_test, T, 0.5)

train_inputs, train_outputs = make_moon_dataset_bin_pop_coding(n_samples_train, T, 0., n_neurons_per_dim)
test_inputs, test_outputs = make_moon_dataset_bin_pop_coding(n_samples_test, T, 0.5, n_neurons_per_dim)


batch_size = 16
input_size = [train_inputs.shape[-1]]
burnin = 10


model = LIFMLP(input_size,
               2,
               n_neurons=[512, 256],
               with_output_layer=False,
               with_bias=False
               ).to(args.device)


# specify loss function

# specify optimizer (stochastic gradient descent) and learning rate = 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
optimizer = torch.optim.Adamax(model.get_trainable_parameters(), lr=1e-2, betas=[0., .95])

criterion = [torch.nn.SmoothL1Loss() for _ in range(model.num_layers)]
# criterion = [one_hot_crossentropy for _ in range(binary_model.num_layers)]

if model.with_output_layer:
    criterion[-1] = one_hot_crossentropy

decolle_loss = DECOLLELoss(criterion, model)

model.init_parameters()

for epoch in range(args.n_epochs):
    loss = 0

    n_batchs = n_samples_train // batch_size + (1 - (n_samples_train % batch_size == 0))
    idx_avail = np.arange(n_samples_train)
    idxs_used = []

    print('Epoch %d/%d' % (epoch, args.n_epochs))
    preds = torch.FloatTensor()
    true_labels = torch.FloatTensor()

    for i in tqdm(range(n_batchs)):
        if (i == (n_batchs - 1)) & (n_samples_train % batch_size != 0):
            batch_size_curr = n_samples_train % batch_size
        else:
            batch_size_curr = batch_size


        idxs = np.random.choice(idx_avail, [batch_size_curr], replace=False)
        idxs_used += list(idxs)
        idx_avail = [i for i in idx_avail if i not in idxs]


        inputs = train_inputs[idxs].permute(1, 0, 2).to(args.device)
        labels = train_outputs[idxs].permute(0, 2, 1).to(args.device)

        # print([Counter(w.detach().numpy().flatten()) for w in binary_model.parameters()])
        model.init(inputs, burnin=burnin)

        readout_hist = [torch.Tensor() for _ in range(len(model.readout_layers))]

        for t in range(burnin, T):
            # forward pass: compute predicted outputs by passing inputs to the model
            s, r, u = model(inputs[t])

            for l, ro_h in enumerate(readout_hist):
                readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

            # calculate the loss
            loss = decolle_loss(s, r, u, target=labels[:, :, t])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            preds = torch.cat((preds, torch.sum(readout_hist[-1], dim=0).argmax(dim=1)))
            true_labels = torch.cat((true_labels, torch.sum(labels.cpu(), dim=-1).argmax(dim=1)))

            # print(u[-1])
            # print(torch.sum(readout_hist[-1], dim=0))
            # print(torch.sum(readout_hist[-1], dim=0).argmax(dim=1))
            # print(torch.sum(labels.cpu(), dim=-1).argmax(dim=1))

    print(true_labels)
    print(preds)
    idxs_wrong = np.where(preds != true_labels)[0]
    idxs_used = np.array(idxs_used)

    # print(torch.sum(train_inputs[np.sort(idxs_used[idxs_wrong])], dim=1)/T)
    # print(np.sort(idxs_used[idxs_wrong]))

    acc = torch.sum(preds == true_labels).float() / n_samples_train
    # backward pass: compute gradient of the loss with respect to model parameters
    print(acc)

    if (epoch + 1) % 10 == 0:
        n_batchs_test = n_samples_test // batch_size + (1 - (n_samples_test % batch_size == 0))
        idx_avail = np.arange(n_samples_test)
        idxs_used = []

        print('Epoch %d/%d' % (epoch, args.n_epochs))
        preds = torch.FloatTensor()
        true_labels = torch.FloatTensor()

        predictions = torch.FloatTensor()
        labels_test = torch.FloatTensor()

        for i in tqdm(range(n_batchs_test)):
            if (i == (n_batchs_test - 1)) & (n_samples_test % batch_size != 0):
                batch_size_curr = n_samples_test % batch_size
            else:
                batch_size_curr = batch_size

            idxs_test = np.random.choice(idx_avail, [batch_size_curr], replace=False)
            idx_avail = [i for i in idx_avail if i not in idxs_test]

            inputs = test_inputs[idxs_test].permute(1, 0, 2).to(args.device)
            labels = test_outputs[idxs_test].permute(0, 2, 1).to(args.device)

            model.init(inputs, burnin=burnin)

            readout_hist = [torch.Tensor() for _ in range(len(model.readout_layers))]

            for t in range(burnin, T):
                # forward pass: compute new pseudo-binary weights
                # forward pass: compute predicted outputs by passing inputs to the model
                s, r, u = model(inputs[t])

                for l, ro_h in enumerate(readout_hist):
                    readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

            predictions = torch.cat((predictions, torch.sum(readout_hist[-1], dim=0).argmax(dim=1)))
            labels_test = torch.cat((labels_test, torch.sum(labels.cpu(), dim=-1).argmax(dim=1)))

        acc = torch.sum(predictions == labels_test).float() / (batch_size * n_batchs_test)

        print('Epoch %d/ test acc %f' % (epoch, acc))
