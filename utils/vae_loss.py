import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm

def sample_reparameterize(mean, std):
    """
    Perform the reparameterization trick to sample from a distribution with the given mean and std
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions
        std - Tensor of arbitrary shape with strictly positive values. Denotes the standard deviation
              of the distribution
    Outputs:
        z - A sample of the distributions, with gradient support for both mean and std. 
            The tensor should have the same shape as the mean and std input tensors.
    """

    #u = torch.normal(0, 1, size=mean.size())
    u = torch.randn_like(mean)
    if torch.min(std) < 0:
        z = torch.exp(std) * u + mean
    else:
        z = std * u + mean

    return z


def KLD(mean, log_std):
    """
    Calculates the Kullback-Leibler divergence of given distributions to unit Gaussians over the last dimension.
    Inputs:
        mean - Tensor of arbitrary shape and range, denoting the mean of the distributions.
        log_std - Tensor of arbitrary shape and range, denoting the log standard deviation of the distributions.
    Outputs:
        kld - Tensor with one less dimension than mean and log_std (summed over last dimension).
              The values represent the Kullback-Leibler divergence to unit Gaussians.
    """

    log_var = 2*log_std
    KLD = 0.5 * torch.sum(torch.exp(log_var) + mean ** 2 - 1 - log_var, dim=1)

    return KLD


def ELBO(input_batch, reconstructed, mean, log_std):
    """
    

    Args:
        input_batch ([type]): [description]
        reconstructed ([type]): [description]
        mean ([type]): [description]
        log_std ([type]): [description]

    Returns:
        [type]: [description]
    """

    # rec_loss = torch.sum(F.binary_cross_entropy(reconstructed, input_batch, reduction='none'),
    #                      dim=(1, 2, 3))
    rec_loss = torch.sum(F.mse_loss(reconstructed, input_batch, reduction='none'),
                         dim=(1, 2, 3))
    reg_loss = KLD(mean, log_std)
    elbo = rec_loss + reg_loss

    return elbo, rec_loss, reg_loss


def ELBO_to_BPD(elbo, batch):
    """
    Converts the summed negative log likelihood given by the ELBO into the bits per dimension score.
    Inputs:
        elbo - Tensor of shape [batch_size]
        batch - Shape of the input images, representing [batch_size, channels, height, width]
    Outputs:
        bpd - The negative log likelihood in bits per dimension for the given image.
    """

    channels = torch.log2(torch.exp(torch.tensor([1.0]))) \
        / torch.prod(torch.tensor(batch[1:]))
    channels = channels.item()

    bpd = elbo * channels

    return bpd
