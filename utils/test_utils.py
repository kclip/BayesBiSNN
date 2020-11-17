import numpy as np
import torch
from data_preprocessing.load_data import get_batch_example
from collections import Counter

def mode_testing_dataset(binary_model, optimizer, burnin, n_examples, batch_size, datagroup, T, labels, input_size, dt, x_max, polarity, device):
    with torch.no_grad():
        optimizer.get_concrete_weights_mode()
        # print([Counter(w.detach().numpy().flatten()) for w in binary_model.parameters()])
        n_batchs = n_examples // batch_size + (1 - (n_examples % batch_size == 0))
        idx_avail = np.arange(n_examples)
        idxs_used = []

        predictions = torch.FloatTensor()

        for i in range(n_batchs):
            if (i == (n_batchs - 1)) & (n_examples % batch_size != 0):
                batch_size_curr = n_examples % batch_size
            else:
                batch_size_curr = batch_size

            idxs = np.random.choice(idx_avail, [batch_size_curr], replace=False)
            idxs_used += list(idxs)
            idx_avail = [i for i in idx_avail if i not in idxs_used]

            inputs, labels = get_batch_example(datagroup, idxs, batch_size_curr, T, labels, input_size, dt, x_max, polarity)
            inputs = inputs.transpose(0, 1).to(device)

            binary_model.init(inputs, burnin=burnin)

            readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

            for t in range(burnin, T):
                # forward pass: compute predicted outputs by passing inputs to the model
                s, r, u = binary_model(inputs[t])

                for l, ro_h in enumerate(readout_hist):
                    readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

            predictions = torch.cat((predictions, readout_hist[-1].transpose(0, 1)))

        return predictions, idxs_used

def mean_testing_dataset(binary_model, optimizer, burnin, n_samples, n_outputs, n_examples, batch_size, datagroup, T, labels, input_size, dt, x_max, polarity, device):
    with torch.no_grad():
        n_batchs = n_examples // batch_size + (1 - (n_examples % batch_size == 0))
        idx_avail = np.arange(n_examples)
        idxs_used = []

        predictions = torch.FloatTensor()

        for i in range(n_batchs):
            if (i == (n_batchs - 1)) & (n_examples % batch_size != 0):
                batch_size_curr = n_examples % batch_size
            else:
                batch_size_curr = batch_size

            idxs = np.random.choice(idx_avail, [batch_size_curr], replace=False)
            idxs_used += list(idxs)
            idx_avail = [i for i in idx_avail if i not in idxs_used]

            inputs, labels = get_batch_example(datagroup, idxs, batch_size_curr, T, labels, input_size, dt, x_max, polarity)
            inputs = inputs.transpose(0, 1).to(device)

            binary_model.init(inputs, burnin=burnin)

            predictions_batch = torch.zeros([batch_size_curr, n_samples, T - burnin, n_outputs])

            for j in range(n_samples):
                optimizer.update_concrete_weights(test=True)
                # print([Counter(w.detach().numpy().flatten()) for w in binary_model.parameters()])

                binary_model.init(inputs, burnin=burnin)

                readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

                for t in range(burnin, T):
                    # forward pass: compute predicted outputs by passing inputs to the model
                    s, r, u = binary_model(inputs[t])

                    for l, ro_h in enumerate(readout_hist):
                        readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

                predictions_batch[:, j] = readout_hist[-1].transpose(0, 1)

            predictions = torch.cat((predictions, predictions_batch))

        return predictions, idxs_used

def mode_testing(binary_model, optimizer, burnin, n_examples, batch_size, data, T, device):
    with torch.no_grad():
        optimizer.get_concrete_weights_mode()
        # print([Counter(w.detach().numpy().flatten()) for w in binary_model.parameters()])
        n_batchs = n_examples // batch_size + (1 - (n_examples % batch_size == 0))
        idx_avail = np.arange(n_examples)
        idxs_used = []

        predictions = torch.FloatTensor()

        for i in range(n_batchs):
            if (i == (n_batchs - 1)) & (n_examples % batch_size != 0):
                batch_size_curr = n_examples % batch_size
            else:
                batch_size_curr = batch_size

            idxs = np.random.choice(idx_avail, [batch_size_curr], replace=False)
            idxs_used += list(idxs)
            idx_avail = [i for i in idx_avail if i not in idxs_used]

            inputs = data[idxs].transpose(0, 1).to(device)

            binary_model.init(inputs, burnin=burnin)

            readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

            for t in range(burnin, T):
                # forward pass: compute predicted outputs by passing inputs to the model
                s, r, u = binary_model(inputs[t])

                for l, ro_h in enumerate(readout_hist):
                    readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

            predictions = torch.cat((predictions, readout_hist[-1].transpose(0, 1)))

        return predictions, idxs_used


def mean_testing(binary_model, optimizer, burnin, n_samples, n_outputs, n_examples, batch_size, data, T, device):
    with torch.no_grad():
        n_batchs = n_examples // batch_size + (1 - (n_examples % batch_size == 0))
        idx_avail = np.arange(n_examples)
        idxs_used = []

        predictions = torch.FloatTensor()

        for i in range(n_batchs):
            if (i == (n_batchs - 1)) & (n_examples % batch_size != 0):
                batch_size_curr = n_examples % batch_size
            else:
                batch_size_curr = batch_size

            idxs = np.random.choice(idx_avail, [batch_size_curr], replace=False)
            idxs_used += list(idxs)
            idx_avail = [i for i in idx_avail if i not in idxs_used]

            inputs = data[idxs].transpose(0, 1).to(device)

            binary_model.init(inputs, burnin=burnin)

            predictions_batch = torch.zeros([batch_size_curr, n_samples, T - burnin, n_outputs])

            for j in range(n_samples):
                optimizer.update_concrete_weights(test=True)
                # print([Counter(w.detach().numpy().flatten()) for w in binary_model.parameters()])

                binary_model.init(inputs, burnin=burnin)

                readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

                for t in range(burnin, T):
                    # forward pass: compute predicted outputs by passing inputs to the model
                    s, r, u = binary_model(inputs[t])

                    for l, ro_h in enumerate(readout_hist):
                        readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

                predictions_batch[:, j] = readout_hist[-1].transpose(0, 1)

            predictions = torch.cat((predictions, predictions_batch))

        return predictions, idxs_used
