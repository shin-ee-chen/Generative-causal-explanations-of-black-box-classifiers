import os

import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
plt.rcParams["text.usetex"]

from utils.vae_loss import sample_reparameterize, ELBO, ELBO_to_BPD

FIGURE_PATH = './Figures'

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
    sweep_range = np.arange(-3, 3+1, 1)
    colors = ['#FFC209', '#0B7ADC', '#8FC839', 'tab:red']

    print('CVAE latent variable sweep for', label)

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
            recon_img = torch.round(torch.sigmoid(model.decoder(z)))
            recon_img.detach_()

            z_sweep.append(recon_img)

            z[:, i] -= z_val

    fig, axes = plt.subplots(1, 1, figsize=(5., 5.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(8, 7),
                     aspect=False
                     )

    for ax, row in zip(grid, z_sweep):
        for n_row, img in enumerate(row):

            t = torch.argmax(model.classifier(img[None, :])).item()

            ax.imshow(img.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='binary',
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
        FIGURE_PATH, model.__class__.__name__ + model.classes_str, str(save_loc))
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, label+'.png'), bbox_inches='tight')

    return fig
