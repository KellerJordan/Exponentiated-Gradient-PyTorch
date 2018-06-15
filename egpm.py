"""
Implementation of EG plus/minus optimizer in PyTorch

Author: Keller Jordan
"""

import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required

class EGPM(Optimizer):

    def __init__(self, params, lr=required,
                 u_scaling=100, norm_per='neuron',
                 weight_regularization=None,
                 gradient_clipping=None,
                 plus_minus=True,
                 init='log_normal'):
        """Initialize a stochastic exponentiated gradient plus/minus optimizer
        
        :param u_scaling: Constant `U` for use in rescaling the sum of positive and negative weights
        after each update
        :param norm_per: 'neuron' | None - determines what set of weights
        to sum over, variable corresponds to an entire layer, neuron sums over the weights going
        into each neuron, weight is only the w_pos and w_neg for each weight, and none does no
        renormalization.
        :param gradient_clipping: None | float - set to None to not use gradient clipping,
        otherwise gradients are clipped to range [-gradient_clipping, +gradient_clipping]
        :param weight_regularization: None | ('entropy', alpha) | ('l1', alpha) - tuple determining
        the type and scale of weight regularization to apply on each update
        :param plus_minus: True | False - whether to use w = w_pos + w_neg weight pairs. If set to
        False, this optimizer will behave as vanilla EG with only w_pos weights
        :param init: 'bootstrap' | 'uniform' | 'log_normal' | 'gamma' - bootstrap initialization sets
        positive and negative weights to preserve old weight values and sets the sum of each
        w_pos + w_neg to equal U, each other option inits the weights such that the mean over each
        neuron is U, and stdev such that the variance of data is preserved as it passes through layer
        """
        
        if norm_per is None:
            u_scaling = 1
        defaults = dict(lr=lr, u_scaling=u_scaling, norm_per=norm_per,
                        gradient_clipping=gradient_clipping,
                        weight_regularization=weight_regularization,
                        plus_minus=plus_minus,
                        init_type=init)
        super().__init__(params, defaults)
        if norm_per not in ['neuron', None]:
            raise Exception('Unknown normalization scheme: per_%s' % norm_per)
        if init not in ['bootstrap', 'uniform', 'log_normal', 'gamma']:
            raise Exception('Unknown initialization scheme: %s' % init)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            u = group['u_scaling']
            norm_per = group['norm_per']
            clip_grad = group['gradient_clipping']
            weight_reg = group['weight_regularization']
            plus_minus = group['plus_minus']
            init_type = group['init_type']
            
            param_layers = self.get_param_layers(group)

            for layer in param_layers:
                
                new_init = self.init_layer(layer, u, init_type, plus_minus)
                
                if not new_init:
                    for p_type, p in layer.items():
                        if clip_grad is not None:
                            p.grad.data.clamp_(-clip_grad, clip_grad)
                        self.update_param(p, lr, p_type, weight_reg)
                
                self.renormalize_layer(layer, u, norm_per)

        return loss

    def get_param_layers(self, group):
        """construct a list of all parameters per layer
        
        ASSUMPTION: bias parameters always follow their respective weights in model.parameters()
        """
        param_layers = []
        
        layer_index = 0
        weights = {}
        biases = {}
        
        for p in group['params']:
            is_bias = len(p.size()) == 1
            
            if is_bias:
                if layer_index in biases:
                    raise Exception('Found two biases not separated by a weight')
                if layer_index not in weights:
                    raise Exception('Found bias that does not follow a weight')
                biases[layer_index] = p
            
            else:
                if layer_index in weights:
                    layer_index += 1
                weights[layer_index] = p

        for i in weights.keys():
            curr_layer = {}
            curr_layer['weight'] = weights[i]
            if i in biases:
                curr_layer['bias'] = biases[i]
            param_layers.append(curr_layer)

        return param_layers

    def init_layer(self, layer, u, init, plus_minus, verbose=False):
        """initialize positive and negative weights for a layer"""

        n_neurons = layer['weight'].size(0)
        n_inputs = (layer['weight'].numel() / n_neurons) + 1
        u_weight = u / n_inputs
        
        if not any('w_pos' not in self.state[p] for p in layer.values()):
            return False
        
        for p in layer.values():
            param_state = self.state[p]

            if plus_minus:
                if init == 'bootstrap':
                    w_pos = (u_weight + p.data.clone()) / 2
                    w_neg = (u_weight - p.data.clone()) / 2
                else:
                    if init == 'uniform':
                        a = u / (2 * n_inputs) - np.sqrt(3 / n_inputs)
                        b = u / (2 * n_inputs) + np.sqrt(3 / n_inputs)
                        assert a > 0
                        dist = torch.distributions.uniform.Uniform(a, b)
                    elif init == 'log_normal':
                        sigmasq = np.log(4 * n_inputs / u**2 + 1)
                        mu = np.log(u) - np.log(2 * n_inputs) - sigmasq / 2
                        dist = torch.distributions.log_normal.LogNormal(mu, np.sqrt(sigmasq))
                    elif init == 'gamma':
                        alpha = u**2 / (4 * n_inputs)
                        beta = u / 2
                        dist = torch.distributions.gamma.Gamma(alpha, beta)
                    w_pos = dist.sample(p.size())
                    w_neg = dist.sample(p.size())

            else:
                if init == 'bootstrap':
                    w_pos = p.data.clone()
                else:
                    raise Exception('Vanilla EG only supports bootstrap initialization')
                w_neg = torch.zeros_like(p)

            param_state['w_pos'] = w_pos
            param_state['w_neg'] = w_neg
            p.data = w_pos - w_neg
        
        if verbose:
            bias_state = self.state[layer['bias']]
            weight_state = self.state[layer['weight']]
            w_pos = torch.cat([bias_state['w_pos'].view(n_neurons, -1), weight_state['w_pos'].view(n_neurons, -1)], 1)
            w_neg = torch.cat([bias_state['w_neg'].view(n_neurons, -1), weight_state['w_neg'].view(n_neurons, -1)], 1)
            sum_mean = (w_pos + w_neg).sum(1).mean().item()
            diff_mean = (w_pos - w_neg).sum(1).mean().item()
            diff_var = (w_pos - w_neg).sum(1).var().item()
            print('Parameter size:', list(w_pos.size()))
            print('Mean sum: %.3f, mean diff: %.3f, variance diff: %.3f, expected var diff: %.3f' %
                  (sum_mean, diff_mean, diff_var, 2))
            
        return True
    
    def update_param(self, p, lr, p_type, weight_reg):
        """update a parameter according to exponentiated gradient update rule,
        does not renormalize. regularization is applied according to weight_reg argument"""
        
        if p.grad is None:
            return
        d_p = p.grad.data
        param_state = self.state[p]

        w_pos = param_state['w_pos']
        w_neg = param_state['w_neg']
        
        # only regularize weights
        if weight_reg is not None and p_type == 'weight':            
            reg_type, alpha = weight_reg
            if reg_type == 'entropy':
                reg_scale = 1 / (1 + alpha)
                r_pos = torch.exp(-lr * d_p * reg_scale)
                r_neg = 1 / r_pos
                update_pos = r_pos * w_pos.pow(reg_scale)
                update_neg = r_neg * w_neg.pow(reg_scale)
            elif reg_type == 'l1':
                reg_scale = np.exp(-lr * alpha)
                r_pos = torch.exp(-lr * d_p)
                r_neg = 1 / r_pos
                update_pos = r_pos * w_pos * reg_scale
                update_neg = r_neg * w_neg * reg_scale
        
        # unregularized update
        else:
            r_pos = torch.exp(-lr * d_p)
            r_neg = 1 / r_pos # torch.exp(-lr * -d_p)
            update_pos = r_pos * w_pos
            update_neg = r_neg * w_neg

        self.state[p]['w_pos'] = update_pos
        self.state[p]['w_neg'] = update_neg
        p.data = update_pos - update_neg

    def renormalize_layer(self, layer, u, norm_per):
        """normalize a layer of the network according to u_scaling and norm_per parameters"""
        
        weights_pos = [self.state[p]['w_pos'] for p in layer.values()]
        weights_neg = [self.state[p]['w_neg'] for p in layer.values()]

        layer_pos = torch.cat([p.view(p.size(0), -1) for p in weights_pos], 1)
        layer_neg = torch.cat([p.view(p.size(0), -1) for p in weights_neg], 1)

        Z = self.normalization(layer_pos + layer_neg, norm_per)
        eps = 1e-7

        for p, w_pos, w_neg in zip(layer.values(), weights_pos, weights_neg):

#             if norm_per == 'weight':
#                 n_outputs = Z.size(0)
#                 n_weights = p.numel()
#                 n_inputs = int(n_weights / n_outputs)
#                 Z_p = Z[:, :n_inputs].clone()
#                 if n_inputs < Z.size(1):
#                     Z = Z[:, n_inputs:]
#                 Z_p = Z_p.view(*p.size())

            if norm_per == 'neuron':
                Z_p = Z.clone()
                while len(Z_p.size()) < len(p.size()):
                    Z_p = Z_p[..., None]

#             elif norm_per == 'variable':
#                 Z_p = Z.clone()
            
            elif norm_per is None:
                Z_p = Z.clone()

            w_pos = self.state[p]['w_pos'] = u * w_pos / (Z_p + eps)
            w_neg = self.state[p]['w_neg'] = u * w_neg / (Z_p + eps)
            p.data = w_pos - w_neg

    def normalization(self, weight_sum, norm_per):
        """get normalizing constant for weight_sum = w_pos + w_neg
        w_neg may be all zero for vanilla EG"""
        
        n_neurons = weight_sum.size(0)
        n_inputs = int(weight_sum.numel() / n_neurons)
        
#         # per_variable: all weights in var get n_neurons * U
#         if norm_per == 'variable':
#             return torch.Tensor([weight_sum.sum() / n_neurons])

        # per_neuron: weight inputs to each neuron get U
        if norm_per == 'neuron':
            update_sum = weight_sum.view(n_neurons, n_inputs)
            return update_sum.sum(1)

#         # per_weight: each weight gets U / n_inputs
#         elif norm_per == 'weight':
#             return weight_sum * n_inputs

        # none: no normalization whatsoever, set u = 1
        elif norm_per is None:
            return torch.Tensor([1.0])
