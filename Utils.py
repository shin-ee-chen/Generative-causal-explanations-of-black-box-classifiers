"""
This file contains some utility functions that can be used throughout training.

"""

import torch
import torch.utils.data as data

import os
import random
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from scipy.stats import norm

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

plt.rcParams["text.usetex"]

CHECKPOINT_PATH = './Models'
FIGURE_PATH = './Figures'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deteministic():
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    
    
def load_latest(trainer, save_name, inference=False, map_location=None):

    def find_latest_version(save_name):
        save_loc = os.path.join(
            CHECKPOINT_PATH, save_name, 'lightning_logs')
        latest_version = os.listdir(save_loc)[-1]
        print(os.path.join(save_loc, latest_version, 'checkpoints'), os.listdir(os.path.join(save_loc, latest_version, 'checkpoints')))
        cpt = os.listdir(os.path.join(
            save_loc, latest_version, 'checkpoints'))[-1]

        return os.path.join(save_loc, latest_version, 'checkpoints', cpt)

    pretrained_filename = find_latest_version(save_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s" %
                pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = trainer.load_from_checkpoint(pretrained_filename)
    else:
        print("No model found")
        return None

    if inference:
        model.eval()
        model.freeze()

    return model
    

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

    #MSE = torch.nn.MSELoss(reduction='none')
    #recon_loss = torch.sum(MSE(input_batch, reconstructed), dim=(1, 2, 3))
    #with torch.no_grad():
    #    print('Input', torch.min(input_batch), torch.max(input_batch))
    #    print('Recon', torch.min(reconstructed), torch.max(reconstructed))
    #    print()
    rec_loss = torch.sum(F.binary_cross_entropy(reconstructed, input_batch, reduction='none'),
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

@torch.no_grad()
def CVAE_sweep(model, i=0, rows=8, dataset=None, save_loc=None):
    """Produces a sweep over a single latent variables in the (C)VAE, 
    and saves the image to the './Figures' directory.

    Args:
        model ([type]): The traind CVAE which is used to explain the classifier.
        i (int): The i-th variable to sweep over.
        rows (int, optional): The number of rows to produce. Defaults to 8.
        dataloader (type, optional): If provided, uses reconstructed images, not random samples. Defaults to None.
            Must be iterable.
        save_loc (path, optional): The name of the save. Defaults to named variable.
    """

    latent_dim = model.K + model.L
    j = i+1
    label = 'alpha_{:d}'.format(
        j) if j <= model.K else 'beta_{:d}'.format(j-model.K)
    sweep_range = range(-3, 3+1, 1)
    colors = ['#FFC209', '#0B7ADC', '#8FC839', 'tab:red']
    
    dataloader = data.DataLoader(dataset, batch_size=rows, shuffle=False,
                                 drop_last=True, pin_memory=True, num_workers=0)
    dataloader = iter(dataloader)

    z_sweep = []
    for r in range(rows):
        if dataloader == None:
            z = torch.normal(0, 1, size=(1, latent_dim))
        
        else:
            try:
                imgs, _ = next(dataloader)
            except StopIteration:
                dataloader = data.DataLoader(dataset, batch_size=rows, shuffle=False,
                                             drop_last=True, pin_memory=True, num_workers=0)
                dataloader = iter(dataloader)
                
                imgs, _ = next(dataloader)
                
            imgs = imgs.to(model.device)
            
            mean, std = model.encoder(imgs[:rows])
            z = sample_reparameterize(mean, std)
        
        z = z.to(model.device)
        
        for z_val in sweep_range:
            
            z[:, i] += z_val
            with torch.no_grad():
                z_sweep.append(torch.round(torch.sigmoid(model.decoder(z))))
            
            z[:, i] -= z_val
            
    fig, axes = plt.subplots(1, 1, figsize=(5., 5.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(8, 7),
                     aspect=False
                     )

    for ax, row in zip(grid, z_sweep):
        for n_row, img in enumerate(row):

            t = torch.argmax(model.classifier(img[None, :])).item()

            ax.imshow(img.permute(1, 2, 0).cpu().numpy(), cmap='binary',
                      vmin=0, vmax=1)

            ax.tick_params(axis='both', which='both',
                           bottom=False, top=False, left=False, right=False,
                           labelbottom=False, labelleft=False)
            for axis in ['top', 'bottom', 'left', 'right']:
                ax.spines[axis].set_linewidth(2.5)
                ax.spines[axis].set_edgecolor(colors[t])

    for axis in ['top', 'left', 'right']:
        axes.spines[axis].set_visible(False)

    axes.tick_params(axis='both', which='both',
                     bottom=True, top=False, left=False, right=False,
                     labelbottom=True, labelleft=False)

    axes.set_xlim(sweep_range.start-0.5, sweep_range.stop-1+0.5)
    axes.set_xticks(list(sweep_range))
    axes.set_xticklabels(['{:+d}'.format(x) for x in sweep_range])
    axes.spines['bottom'].set_position(('outward', 10))

    grid.set_axes_pad([0.05, 0.05])
    fig.suptitle('Sweep over $\\{:s}$'.format(label), y=0.95)
    #fig.tight_layout()

    #if save_loc == None:
    #    save_loc = label

    fig_dir = os.path.join(
        FIGURE_PATH, model.__class__.__name__, str(save_loc))
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, label+'.png'), bbox_inches='tight')

    return fig
