import numpy as np
import os
import torch

from optimizer.BBSNN import BayesBiSNNRP
from optimizer.STBiSNN import BiSGD


def launch_tests(binary_model, optimizer, burnin, train_dl, test_dl, T, epoch, params, device, results_path, output=-1):
    # Wrapper to launch tests
    if isinstance(optimizer, BiSGD):
        launch_test_stbisnn(binary_model, optimizer, burnin, train_dl, test_dl, T, epoch, params, device, results_path, output)
    elif isinstance(optimizer, BayesBiSNNRP):
        launch_test_bbisnn(binary_model, optimizer, burnin, train_dl, test_dl, T, epoch, params, device, results_path, output)
    else:
        raise NotImplementedError

def launch_test_stbisnn(binary_model, optimizer, burnin, train_dl, test_dl, T, epoch, params, device, results_path, output=-1):
    if train_dl is not None:
        print('Testing on train data epoch %d/%d' % (epoch + 1, params['n_epochs']))
        test_stbisnn(binary_model, optimizer, burnin, train_dl, T, epoch, params, device, results_path, 'train', output)

    if test_dl is not None:
        print('Testing on test data epoch %d/%d' % (epoch + 1, params['n_epochs']))
        test_stbisnn(binary_model, optimizer, burnin, test_dl, T, epoch, params, device, results_path, 'test', output)



def launch_test_bbisnn(binary_model, optimizer, burnin, train_dl, test_dl, T, epoch, params, device, results_path, output=-1):
    if train_dl is not None:
        print('Mode testing on train data epoch %d/%d' % (epoch + 1, params['n_epochs']))
        mode_testing(binary_model, optimizer, burnin, iter(train_dl), T, device, results_path, 'train', output)

        print('Mean testing on train data epoch %d/%d' % (epoch + 1, params['n_epochs']))
        mean_testing(binary_model, optimizer, burnin, params['n_samples'], len(params['classes']), iter(train_dl), T, device, results_path, 'train', output)

    if test_dl is not None:
        print('Mode testing on test data epoch %d/%d' % (epoch + 1, params['n_epochs']))
        mode_testing(binary_model, epoch, optimizer, burnin, iter(test_dl), T, device, results_path, 'test', output)

        print('Mean testing on test data epoch %d/%d' % (epoch + 1, params['n_epochs']))
        mean_testing(binary_model, epoch, optimizer, burnin, params['n_samples'], len(params['classes']), iter(test_dl), T, device, results_path, 'test', output)


def mode_testing(binary_model, epoch, optimizer, burnin, iterator, T, device, results_path, data_name, output=-1):
    # MAP testing for Bayes-BiSNN
    with torch.no_grad():
        # Generate MAP binary weights
        optimizer.update_binary_weights_map()

        # Placeholders to record the order of labels (useful if shuffle is True in the dataloader)
        predictions = torch.FloatTensor()
        true_labels = torch.FloatTensor()


        for inputs, labels in iterator:
            inputs = inputs.transpose(0, 1).to(device)

            # Initialize network state
            binary_model.init(inputs, burnin=burnin)

            # Placeholders for the readout layers outputs
            readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

            for t in range(burnin, T):
                # Forward pass
                s, r, u = binary_model(inputs[t])

                # Record readouts outputs
                for l, ro_h in enumerate(readout_hist):
                    readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

            # Record network outputs (from the desired readout layer given by output) and true labels
            predictions = torch.cat((predictions, readout_hist[output].transpose(0, 1)))
            if len(labels.shape) == 3:
                true_labels = torch.cat((true_labels, torch.sum(labels.cpu(), dim=-1).argmax(dim=1).type_as(true_labels)))
            else:
                true_labels = torch.cat((true_labels, labels.type_as(true_labels)))

    np.save(os.path.join(results_path, data_name + '_predictions_latest_mode_epoch_%d' % epoch), predictions.numpy())
    np.save(os.path.join(results_path, data_name + '_true_labels_mode_epoch_%d' % epoch), true_labels.numpy())


def mean_testing(binary_model, epoch, optimizer, burnin, n_samples, n_classes, iterator, T, device, results_path, data_name, output=-1):
    # Ensemble testing for Bayes-BiSNN
    with torch.no_grad():
        # Placeholders to record the order of labels (useful if shuffle is True in the dataloader)
        predictions = torch.FloatTensor()
        true_labels = torch.FloatTensor()

        for inputs, labels in iterator:
            inputs = inputs.transpose(0, 1).to(device)

            # Temporary tensor to record network outputs in the current batch, shape is [batch_size, num MC samples, example length, num of outputs neurons]
            if len(labels.shape) == 3:
                predictions_batch = torch.zeros([inputs.shape[1], n_samples, T - burnin, labels.shape[1]])
            else:
                predictions_batch = torch.zeros([inputs.shape[1], n_samples, T - burnin, n_classes])

            # Repeat the prediction task n_samples times
            for j in range(n_samples):
                # Sample new binary weights following the Bernoulli distribution
                optimizer.update_binary_weights(test=True)

                # Initialize network state
                binary_model.init(inputs, burnin=burnin)

                # Placeholders for the readout layers outputs
                readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

                for t in range(burnin, T):
                    # Forward pass
                    s, r, u = binary_model(inputs[t])

                    # Record readouts outputs
                    for l, ro_h in enumerate(readout_hist):
                        readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

                # Record current realization of the predictions
                predictions_batch[:, j] = readout_hist[output].transpose(0, 1)

            # Record network batch outputs (from the desired readout layer given by output) and true labels
            predictions = torch.cat((predictions, predictions_batch))
            if len(labels.shape) == 3:
                true_labels = torch.cat((true_labels, torch.sum(labels.cpu(), dim=-1).argmax(dim=1).type_as(true_labels)))
            else:
                true_labels = torch.cat((true_labels, labels.type_as(true_labels)))

    np.save(os.path.join(results_path, data_name + '_predictions_latest_mean_epoch_%d' % epoch), predictions.numpy())
    np.save(os.path.join(results_path, data_name + '_true_labels_mean_epoch_%d' % epoch), true_labels.numpy())


def test_stbisnn(binary_model, optimizer, burnin, dataloader, T, epoch, params, device, results_path, data_name, output=-1):
    # Testing for ST-BiSNN
    with torch.no_grad():
        # Generate binary weights
        optimizer.update_binary_weights()

        print('Testing epoch %d/%d' % (epoch + 1, params['n_epochs']))
        # Placeholders to record the order of labels (useful if shuffle is True in the dataloader)
        predictions = torch.FloatTensor()
        true_labels = torch.FloatTensor()

        iterator = iter(dataloader)

        for inputs, labels in iterator:
            inputs = inputs.transpose(0, 1).to(device)

            # Initialize network state
            binary_model.init(inputs, burnin=burnin)

            # Placeholders for the readout layers outputs
            readout_hist = [torch.Tensor() for _ in range(len(binary_model.readout_layers))]

            for t in range(burnin, T):
                # Forward pass
                s, r, u = binary_model(inputs[t])

                # Record readouts outputs
                for l, ro_h in enumerate(readout_hist):
                    readout_hist[l] = torch.cat((ro_h, r[l].cpu().unsqueeze(0)), dim=0)

            # Record network outputs (from the desired readout layer given by output) and true labels
            predictions = torch.cat((predictions, readout_hist[output].transpose(0, 1)))
            if len(labels.shape) == 3:
                true_labels = torch.cat((true_labels, torch.sum(labels.cpu(), dim=-1).argmax(dim=1).type_as(true_labels)))
            else:
                true_labels = torch.cat((true_labels, labels.type_as(true_labels)))

        np.save(os.path.join(results_path, data_name + '_predictions_latest_epoch_%d' % epoch), predictions.numpy())
        np.save(os.path.join(results_path, data_name + '_true_labels_epoch_%d' % epoch), true_labels.numpy())
