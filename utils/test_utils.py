import numpy as np
import torch
from data_preprocessing.load_data import get_batch_example
import os
from collections import Counter

def launch_tests(binary_model, optimizer, burnin, n_examples_test, n_examples_train, test_data, train_data, T, input_size, dt, epoch, args, results_path, output=-1):
    if train_data is not None:
        print('Mode testing on train data epoch %d/%d' % (epoch + 1, args.n_epochs))
        mode_testing_dataset(binary_model, optimizer, burnin, n_examples_train, args.batch_size, train_data,
                             T, args.labels, input_size, dt, 26, args.polarity, args.device, results_path, 'train', output)

        print('Mean testing on train data epoch %d/%d' % (epoch + 1, args.n_epochs))
        mean_testing_dataset(binary_model, optimizer, burnin, args.n_samples, len(args.labels), n_examples_train,
                             args.batch_size, train_data, T, args.labels, input_size, dt, 26, args.polarity, args.device, results_path, 'train', output)

    if test_data is not None:
        print('Mode testing on test data epoch %d/%d' % (epoch + 1, args.n_epochs))
        mode_testing_dataset(binary_model, optimizer, burnin, n_examples_test, args.batch_size, test_data,
                             T, args.labels, input_size, dt, 26, args.polarity, args.device, results_path, 'test', output)

        print('Mean testing on test data epoch %d/%d' % (epoch + 1, args.n_epochs))
        mean_testing_dataset(binary_model, optimizer, burnin, args.n_samples, len(args.labels), n_examples_test,
                             args.batch_size, test_data, T, args.labels, input_size, dt, 26, args.polarity, args.device, results_path, 'test', output)



def mode_testing_dataset(binary_model, optimizer, burnin, n_examples, batch_size, datagroup, T,
                         labels, input_size, dt, x_max, polarity, device, results_path, data_name, output=-1):
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

            inputs, _ = get_batch_example(datagroup, idxs, batch_size_curr, T, labels, input_size, dt, x_max, polarity)
            inputs = inputs.transpose(0, 1).to(device)

            binary_model.init(inputs, burnin=burnin)

            readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

            for t in range(burnin, T):
                # forward pass: compute predicted outputs by passing inputs to the model
                s, r, u = binary_model(inputs[t])

                for l, ro_h in enumerate(readout_hist):
                    readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

            predictions = torch.cat((predictions, readout_hist[output].transpose(0, 1)))

    np.save(os.path.join(results_path, data_name + '_predictions_latest_mode'), predictions.numpy())
    np.save(os.path.join(results_path, data_name + '_idxs_mode'), np.array(idxs_used))


def mean_testing_dataset(binary_model, optimizer, burnin, n_samples, n_outputs, n_examples, batch_size,
                         datagroup, T, labels, input_size, dt, x_max, polarity, device, results_path, data_name,  output=-1):
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

            inputs, _ = get_batch_example(datagroup, idxs, batch_size_curr, T, labels, input_size, dt, x_max, polarity)
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

                predictions_batch[:, j] = readout_hist[output].transpose(0, 1)

            predictions = torch.cat((predictions, predictions_batch))

    np.save(os.path.join(results_path, data_name + '_predictions_latest_mean'), predictions.numpy())
    np.save(os.path.join(results_path, data_name + '_idxs_mean'), np.array(idxs_used))

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
