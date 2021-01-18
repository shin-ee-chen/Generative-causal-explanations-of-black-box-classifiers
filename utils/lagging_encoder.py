"""
Useful functions for sampling and calculating loss of the aggressive trained text-VAE.

Adapted from: https://github.com/jxhe/vae-lagging-encoder/blob/master/modules/utils.py
"""

import torch


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def generate_grid(zmin, zmax, dz, device, ndim=2):
    """generate a 1- or 2-dimensional grid
    Returns: Tensor, int
        Tensor: The grid tensor with shape (k^2, 2),
            where k=(zmax - zmin)/dz
        int: k
    """

    if ndim == 2:
        x = torch.arange(zmin, zmax, dz)
        k = x.size(0)

        x1 = x.unsqueeze(1).repeat(1, k).view(-1)
        x2 = x.repeat(k)

        return torch.cat((x1.unsqueeze(-1), x2.unsqueeze(-1)), dim=-1).to(device), k

    elif ndim == 1:
        return torch.arange(zmin, zmax, dz).unsqueeze(1).to(device)


class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
