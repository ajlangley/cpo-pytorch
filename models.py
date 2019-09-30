import itertools
import torch
from torch.nn import Linear, LogSoftmax, Module, Parameter, Sequential, Tanh
from torch.distributions import Independent
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

from torch_utils.torch_utils import get_device


class MultinomialLayer(Module):
    '''
    Implements a layer that outputs a multinomial distribution

    Methods
    ------
    __call__(log_action_probs)
        Takes as input log probabilities and outputs a pytorch multinomail
        distribution
    '''

    def __init__(self):
        Module.__init__(self)

    def __call__(self, log_action_probs):
        return Categorical(logits=log_action_probs)


class DiagGaussianLayer(Module):
    '''
    Implements a layer that outputs a Gaussian distribution with a diagonal
    covariance matrix

    Attributes
    ----------
    log_std : torch.FloatTensor
        the log square root of the diagonal elements of the covariance matrix

    Methods
    -------
    __call__(mean)
        takes as input a mean vector and outputs a Gaussian distribution with
        diagonal covariance matrix defined by log_std

    '''

    def __init__(self, output_dim=None, log_std=None):
        Module.__init__(self)

        self.log_std = log_std

        if log_std is None:
            self.log_std = Parameter(torch.zeros(output_dim), requires_grad=True)

    def __call__(self, mean):
        std = torch.exp(self.log_std)
        normal_dist = Independent(Normal(loc=mean, scale=std), 1)

        return normal_dist


def build_layers(input_dim, hidden_dims, output_dim):
    '''
    Returns a list of Linear and Tanh layers with the specified layer sizes

    Parameters
    ----------
    input_dim : int
        the input dimension of the first linear layer

    hidden_dims : list
        a list of type int specifying the sizes of the hidden layers

    output_dim : int
        the output dimension of the final layer in the list

    Returns
    -------
    layers : list
        a list of Linear layers, each one followed by a Tanh layer, excluding the
        final layer
    '''

    layer_sizes = [input_dim] + hidden_dims + [output_dim]
    layers = []

    for i in range(len(layer_sizes) - 1):
        layers.append(Linear(layer_sizes[i], layer_sizes[i + 1], bias=True))

        if i != len(layer_sizes) - 2:
            layers.append(Tanh())

    return layers

def build_mlp(input_dim, hidden_dims, output_dim):
    '''
    Build a multilayer perceptron with tanh activations with the specified input,
    output, and hidden layer sizes

    Parameters
    ----------
    input_dim : int
        the input dimension of the first linear layer

    hidden_dims : list
        a list of type int specifying the sizes of the hidden layers

    output_dim : int
        the output dimension of the final layer in the list

    Returns
    -------
    mlp : torch.nn.Sequential
        a pytorch sequential model that defines a MLP
    '''

    mlp = Sequential(*build_layers(input_dim, hidden_dims, output_dim))

    return mlp

def build_diag_gauss_policy(state_dim, hidden_dims, action_dim,
    log_std=None):
    '''
    Build a multilayer perceptron with a DiagGaussianLayer at the output layer

    Parameters
    ----------
    state_dim : int
        the input size of the network

    hidden_dims : list
        a list of type int specifying the sizes of the hidden layers

    action_dim : int
        the dimensionality of the Gaussian distribution to be outputted by the
        policy

    log_std : torch.FloatTensor
        the log square root of the diagonal elements of the covariance matrix
        (will be set to a vector of zeros if none is specified)

    Returns
    -------
    policy : torch.nn.Sequential
        a pytorch sequential model that outputs a Gaussian distribution
    '''

    layers = build_layers(state_dim, hidden_dims, action_dim)
    layers[-1].weight.data *= 0.1
    layers[-1].bias.data *= 0.0
    layers.append(DiagGaussianLayer(action_dim, log_std))
    policy = Sequential(*layers)

    return policy
