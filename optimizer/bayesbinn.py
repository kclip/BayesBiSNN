import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import numpy as np

"""
################################
BayesBiSNN optimizer
Adapted from https://github.com/team-approx-bayes/BayesBiNN/
################################

"""


required = object()


def update_input(self, input, output):
    self.input = input[0].data
    self.output = output


class BayesBiNN(Optimizer):
    """BayesBiNN. It uses the mean-field Bernoulli approximation. Note that currently this
        optimizer does **not** support multiple model parameter groups. All model
        parameters must use the same optimizer parameters.
        model (nn.Module): network model
        train_set_size (int): number of data samples in the full training set
        lr (float, optional): learning rate
        betas (float, optional): coefficient used for computing
            running average of gradients
        prior_w_r (FloatTensor, optional): w_r of prior distribution (posterior of previous task)
            (default: None)
        num_samples (float, optional): number of MC samples
            (default: 1), if num_samples=0, we just use the point estimate mu instead of sampling
        temperature (float): temperature value of the Gumbel soft-max trick
        reweight: reweighting scaling factor of the KL term
    """

    def __init__(self, model, train_set_size, lr=1e-9, betas=0.0, prior_w_r=None, num_samples=5, w_r_init=10, w_r_std=0, temperature=1, reweight=1):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if prior_w_r is not None and not torch.is_tensor(prior_w_r):
            raise ValueError("Invalid prior mu value (from previous task): {}".format(prior_w_r))

        if not 0.0 <= betas < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas))
        if train_set_size < 1:
            raise ValueError("Invalid number of training data points: {}".format(train_set_size))

        defaults = dict(lr=lr, beta=betas,
                        prior_w_r=prior_w_r, num_samples=num_samples,
                        train_set_size=train_set_size, temperature=temperature, reweight=reweight)

        # only support a single parameter group.
        super(BayesBiNN, self).__init__(model.parameters(), defaults)

        self.train_modules = []
        self.set_train_modules(model)  # to obtain the trained modules in the model

        # We only support a single parameter group
        parameters = self.param_groups[0]['params']  # self.param_groups is a self-contained parameter group inside optimizer

        self.param_groups[0]['lr'] = lr

        device = parameters[0].device
        p = parameters_to_vector(parameters)

        # natural parameter of Bernoulli distribution.
        mixtures_coeff = torch.randint_like(p, 2)
        self.state['w_r'] = mixtures_coeff * (w_r_init + np.sqrt(w_r_std) * torch.randn_like(p)) \
                              + (1 - mixtures_coeff) * (-w_r_init + np.sqrt(w_r_std) * torch.randn_like(p))   # such initialization is empirically good, others are OK of course

        # expectation parameter of Bernoulli distribution.
        self.state['mu'] = torch.tanh(self.state['w_r'])

        # momentum term
        self.state['momentum'] = torch.zeros_like(p, device=device)  # momentum

        # expectation parameter of prior distribution.
        if torch.is_tensor(self.defaults['prior_w_r']):
            self.state['prior_w_r'] = self.defaults['prior_w_r'].to(device)
        else:
            self.state['prior_w_r'] = torch.zeros_like(p, device=device)

        # step initialization
        self.state['step'] = 0
        self.state['temperature'] = temperature
        self.state['reweight'] = reweight




    def set_train_modules(self, module):
        if len(list(module.children())) == 0:
            if len(list(module.parameters())) != 0:
                self.train_modules.append(module)
        else:
            for child in list(module.children()):
                self.set_train_modules(child)




    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss without doing the backward pass
        """
        if closure is None:
            raise RuntimeError(
                'For now, BayesBiNN only supports that the model/loss can be reevaluated inside the step function')

        self.state['step'] += 1

        # We only support a single parameter group.
        parameters = self.param_groups[0]['params']
        grad_hat = torch.zeros_like(self.state['w_r'])

        loss_list = []
        pred_list = []

        if self.defaults['num_samples'] <= 0:
            # Simply using the point estimate mu instead of sampling
            w_vector = torch.tanh(self.state['w_r'])
            vector_to_parameters(w_vector, parameters)

            # Get loss and predictions
            loss, preds = closure()

            pred_list.append(preds)

            linear_grad = torch.autograd.grad(loss, parameters)  # compute the gradients over the
            loss_list.append(loss.detach())

            grad = parameters_to_vector(linear_grad).detach()

            grad_hat = self.defaults['train_set_size'] * grad

        else:
            # Using Monte Carlo samples
            for _ in range(self.defaults['num_samples']):  # sampling samples to estimate the gradients
                # Sample a parameter vector:
                raw_noise = torch.rand_like(self.state['mu'])

                delta = torch.log(raw_noise / (1 - raw_noise)) / 2

                w_b_concrete = torch.tanh((delta + self.state['w_r']) / self.defaults['temperature'])

                vector_to_parameters(w_b_concrete, parameters)

                # Get loss and predictions
                loss, preds = closure()

                pred_list.append(preds)

                linear_grad = torch.autograd.grad(loss, parameters)  # compute the gradients over the parameters
                loss_list.append(loss.detach())

                # Convert the parameter gradient to a single vector.
                grad = parameters_to_vector(linear_grad).detach()

                scale = (1 - w_b_concrete * w_b_concrete + 1e-10) / self.defaults['temperature'] / (1 - self.state['mu'] * self.state['mu'] + 1e-10)
                grad_hat.add_(scale * grad)

            grad_hat = grad_hat.mul(self.defaults['train_set_size'] / self.defaults['num_samples'])

        # Add momentum
        self.state['momentum'] = self.defaults['beta'] * self.state['momentum'] \
                                 + (1 - self.defaults['beta']) * (grad_hat + self.state['reweight'] * (self.state['w_r'] - self.state['prior_w_r']))

        # Get the mean loss over the number of samples
        loss = torch.mean(torch.stack(loss_list))

        # Bias correction of momentum as adam
        bias_correction1 = 1 - self.defaults['beta'] ** self.state['step']

        # Update w_r vector
        self.state['w_r'] = self.state['w_r'] - self.param_groups[0]['lr'] * self.state['momentum'] / bias_correction1
        self.state['mu'] = torch.tanh(self.state['w_r'])

        return loss, pred_list



    def get_distribution_params(self):
        """Returns current mean and precision of variational distribution
           (usually used to save parameters from current task as prior for next task).
        """
        mu = self.state['mu'].clone().detach()
        precision = mu * (1 - mu)  # variance term

        return mu, precision

    def get_mc_predictions(self, forward_function, inputs, ret_numpy=False, raw_noises=None, *args, **kwargs):
        """Returns Monte Carlo predictions.
        Arguments:
            forward_function (callable): The forward function of the model
                that takes inputs and returns the outputs.
            inputs (FloatTensor): The inputs to the model.
            mc_samples (int): The number of Monte Carlo samples.
            ret_numpy (bool): If true, the returned list contains numpy arrays,
                otherwise it contains torch tensors.
        """

        # We only support a single parameter group.
        parameters = self.param_groups[0]['params']
        predictions = []

        if raw_noises is None:  # use the mean value (sign) to make predictions
            raw_noises = []
            mean_vector = torch.where(self.state['mu'] <= 0, torch.zeros_like(self.state['mu']), torch.ones_like(self.state['mu']))
            raw_noises.append(mean_vector)  # perform inference using the sign of the mean value when there is no sampling

        for raw_noise in raw_noises:
            # Sample a parameter vector:
            vector_to_parameters(2 * raw_noise - 1, parameters)
            # Call the forward computation function
            outputs = forward_function(inputs, *args, **kwargs)
            if ret_numpy:
                outputs = outputs.data.cpu().numpy()
            predictions.append(outputs)

        return predictions
