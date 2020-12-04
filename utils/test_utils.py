import numpy as np
import torch
from data_preprocessing.load_data_old import get_batch_example
import os
from collections import Counter

def launch_tests(binary_model, optimizer, burnin, train_iterator, test_iterator, T, epoch, args, results_path, output=-1):
    if train_iterator is not None:
        print('Mode testing on train data epoch %d/%d' % (epoch + 1, args.n_epochs))
        mode_testing_dataset(binary_model, optimizer, burnin, train_iterator, T, args.device, results_path, 'train', output)

        print('Mean testing on train data epoch %d/%d' % (epoch + 1, args.n_epochs))
        mean_testing_dataset(binary_model, optimizer, burnin, args.n_samples, len(args.classes), train_iterator, T, args.device, results_path, 'train', output)

    if test_iterator is not None:
        print('Mode testing on test data epoch %d/%d' % (epoch + 1, args.n_epochs))
        mode_testing_dataset(binary_model, optimizer, burnin, test_iterator, T, args.device, results_path, 'test', output)

        print('Mean testing on test data epoch %d/%d' % (epoch + 1, args.n_epochs))
        mean_testing_dataset(binary_model, optimizer, burnin, args.n_samples, len(args.classes), test_iterator, T, args.device, results_path, 'test', output)



def mode_testing_dataset(binary_model, optimizer, burnin, iterator, T, device, results_path, data_name, output=-1):
    with torch.no_grad():
        optimizer.get_concrete_weights_mode()

        predictions = torch.FloatTensor()
        true_labels = torch.FloatTensor()

        for inputs, labels in iterator:
            inputs = inputs.transpose(0, 1).to(device)

            binary_model.init(inputs, burnin=burnin)

            readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

            for t in range(burnin, T):
                # forward pass: compute predicted outputs by passing inputs to the model
                s, r, u = binary_model(inputs[t])

                for l, ro_h in enumerate(readout_hist):
                    readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

            predictions = torch.cat((predictions, readout_hist[output].transpose(0, 1)))
            true_labels = torch.cat((true_labels, torch.sum(labels.cpu(), dim=-1).argmax(dim=1).type_as(true_labels)))

    np.save(os.path.join(results_path, data_name + '_predictions_latest_mode'), predictions.numpy())
    np.save(os.path.join(results_path, data_name + '_true_labels_mode'), true_labels.numpy())


def mean_testing_dataset(binary_model, optimizer, burnin, n_samples, n_outputs, iterator, T, device, results_path, data_name, output=-1):
    with torch.no_grad():

        predictions = torch.FloatTensor()
        true_labels = torch.FloatTensor()

        for inputs, labels in iterator:
            inputs = inputs.transpose(0, 1).to(device)

            binary_model.init(inputs, burnin=burnin)

            predictions_batch = torch.zeros([inputs.shape[0], n_samples, T - burnin, n_outputs])

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

                predictions_batch[:, j] = readout_hist[output].transpose(0, 1)

            predictions = torch.cat((predictions, predictions_batch))
            true_labels = torch.cat((true_labels, torch.sum(labels.cpu(), dim=-1).argmax(dim=1).type_as(true_labels)))

    np.save(os.path.join(results_path, data_name + '_predictions_latest_mean'), predictions.numpy())
    np.save(os.path.join(results_path, data_name + '_true_labels_mean'), true_labels.numpy())


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
