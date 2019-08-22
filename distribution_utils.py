import torch
from torch.distributions import Independent
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal


def detach_dist(dist):
    '''
    Return a copy of dist with the distribution parameters detached from the
    computational graph

    Parameters
    ----------
    dist: torch.distributions.distribution.Distribution
        the distribution object for which the detached copy is to be returned

    Returns
    -------
    detached_dist
        the detached distribution
    '''

    if type(dist) is Categorical:
        detached_dist = Categorical(logits=dist.logits.detach())
    elif type(dist) is Independent:
        detached_dist = Normal(loc=dist.mean.detach(), scale=dist.stddev.detach())
        detached_dist = Independent(detached_dist, 1)

    return detached_dist

def mean_kl_first_fixed(dist_1, dist_2):
    '''
    Calculate the kl-divergence between dist_1 and dist_2 after detaching dist_1
    from the computational graph

    Parameters
    ----------
    dist_1 : torch.distributions.distribution.Distribution
        the first argument to the kl-divergence function (will be fixed)

    dist_2 : torch.distributions.distribution.Distribution
        the second argument to the kl-divergence function (will not be fixed)

    Returns
    -------
    mean_kl : torch.float
        the kl-divergence between dist_1 and dist_2
    '''
    dist_1_detached = detach_dist(dist_1)
    mean_kl = torch.mean(kl_divergence(dist_1_detached, dist_2))

    return mean_kl
