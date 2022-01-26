import torch
from torch.optim.optimizer import _RequiredParameter, Optimizer
from copy import deepcopy
required = _RequiredParameter()


class BiOptimizer(torch.optim.Optimizer):
    def __init__(self, binary_params, latent_params, defaults):
        super(BiOptimizer, self).__init__(latent_params, defaults)

        self.binary_param_groups = []
        binary_param_groups = list(binary_params)

        if len(binary_param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(binary_param_groups[0], dict):
            binary_param_groups = [{'params': binary_param_groups}]

        for binary_param_group in binary_param_groups:
            self.add_binary_param_group(binary_param_group)


    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
            'binary_param_groups': self.binary_param_groups
        }


    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])

        for i, binary_group in enumerate(self.binary_param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(binary_group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, binary_group[key])

        format_string += ')'
        return format_string


    def zero_grad(self):
        r"""Clears the gradients of all optimized :class:`torch.Tensor` s."""
        for group in self.binary_param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()


    def add_binary_param_group(self, binary_param_group):
        r"""Add a binary param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Arguments:
            param_group (dict): Specifies what Tensors should be optimized along with group
            specific optimization options.
        """

        assert isinstance(binary_param_group, dict), "param group must be a dict"

        binary_params = binary_param_group['params']
        if isinstance(binary_params, torch.Tensor):
            binary_param_group['params'] = [binary_params]
        elif isinstance(binary_params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            binary_param_group['params'] = list(binary_params)

        for param in binary_param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + torch.typename(param))
            if not param.is_leaf:
                raise ValueError("can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            if default is required and name not in binary_param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                binary_param_group.setdefault(name, default)

        param_set = set()
        for group in self.binary_param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(binary_param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.binary_param_groups.append(binary_param_group)




