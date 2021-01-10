import torch
import torch.nn as nn
import numpy as np

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from MNIST import MNIST_limited
from Utils import *
from InformationFlow import *

from PIL import Image

EPS = 1e-8

class Encoder(nn.Module):
    def __init__(self, img_channels: int = 1, num_filters: int = 64,
                 latent_dim: int = 8):
        """
        Encoder with a CNN network. Defaults to the OShaugnessy architecture in Table 2, Appendix E

        Inputs:
            img_channels - Number of input channels of the image.
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            latent_dim - The number of latents to encode into. Note, this is K+L.
        """
        super().__init__()

        c_hid = num_filters
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, num_filters, kernel_size=4,
                      padding=1, stride=2),  # 28x28 => 14x14
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=4,
                      padding=1, stride=2),  # 14x14 => 7x7 
            nn.ReLU(),
            nn.Conv2d(num_filters, num_filters, kernel_size=4,
                      padding=0, stride=1),  # 7x7 => 4x4
            nn.ReLU(),
            nn.Flatten(),  # Image grid to single feature vector
        )
        
        self.mean = nn.Linear(4*4*num_filters, latent_dim)
        self.log_std = nn.Linear(4*4*num_filters, latent_dim)
        
        for layer in [self.net.children()] + [self.mean] + [self.log_std]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, imgs):
        """
        Inputs:
            imgs - Input batch with images of shape [B,C,H,W] and range 0 to 1.
        Outputs:
            mean - Tensor of shape [B,z_dim] representing the predicted mean of the latent distributions.
            log_std - Tensor of shape [B,z_dim] representing the predicted log standard deviation
                      of the latent distributions.
        """

        hidden = self.net(imgs)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden)

        return mean, log_std
    

class Decoder(nn.Module):
    def __init__(self, img_channels: int = 1, num_filters: int = 64,
                 latent_dim: int = 8):
        """
        Decoder with a CNN network. Defaults to the OShaugnessy architecture in Table 2, Appendix E

        Inputs:
            img_channels - Number of input channels of the image.
            num_filters - Number of channels we use in the first convolutional
                          layers. Deeper layers might use a duplicate of it.
            latent_dim - The number of latents to encode into. Note, this is K+L.
        """
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 3136),
            nn.ReLU()
        )
        
        self.convnet = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=4,
                      padding=1, stride=1),  # 7x7 => 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, num_filters, kernel_size=4,
                      padding=2, stride=2),  # 8x8 => 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(num_filters, img_channels, kernel_size=4,
                      padding=1, stride=2),  # 14x14 => 28x28
        )
        
        for layer in [self.linear.children()] + [self.convnet.children()]:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, z):
        """
        Inputs:
            z - Latent vector of shape [B,z_dim]
        Outputs:
            x - Prediction of the reconstructed image based on z. Shape: [B,num_input_channels,28,28]
        """

        x = self.linear(z)
        x = x.reshape(x.shape[0], -1, 7, 7)
        x = self.convnet(x)

        return x

    @property
    def device(self):
        """
        Property function to get the device on which the decoder is.
        Might be helpful in other functions.
        """
        return next(self.parameters()).device


class MNIST_VAE(pl.LightningModule):

    def __init__(self, classifier_name, num_filters, K=1, L=7, M=2, lamb=0.05, lr=1e-3, Nalpha=10, Nbeta=10, use_ce=True):
        """
        PyTorch Lightning module that summarizes all components to train a VAE.
        Inputs:
            classifier_name - String denoting what classifier black-box to use.
            num_filters - Number of channels to use in a CNN encoder/decoder
            K - Number of causal latent variables
            L - Number of non-causal latent variables
            M - Number of classifier classes
            lamb - The lambda coefficient controlling how much influence the VAE loss has
            lr - Learning rate to use for the optimizer
            Nalpha - Number of samples used to approximate the information flow loss
            Nbeta - Same as above, but then for the non-causal latent variables
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.K = K
        self.L = L
        self.M = M
        self.lamb = lamb
        
        self.latent_dim = self.K + self.L
                    
        self.Nalpha = Nalpha
        self.Nbeta = Nbeta

        self.encoder = Encoder(img_channels=1, num_filters=num_filters, latent_dim=K+L)
        self.decoder = Decoder(img_channels=1, num_filters=num_filters, latent_dim=K+L)
        
        if classifier_name == 'MNIST_CNN_OShaugnessy':
            from CNN_MNIST_trainer import MNIST_CNN_OShaugnessy
            self.classifier = load_latest(MNIST_CNN_OShaugnessy, 'MNIST_CNN', 
                                          inference=True, map_location=self.device)
        else:
            print('No classifier specified')
            return None
        
        self.use_ce = 0 if use_ce == False else 1


    def forward(self, imgs):
        """
        The forward function calculates the VAE-loss for a given batch of images.
        Inputs:
            imgs - Batch of images of shape [B,C,H,W]
        Ouptuts:
            L_rec - The average reconstruction loss of the batch. Shape: single scalar
            L_reg - The average regularization loss (KLD) of the batch. Shape: single scalar
            bpd - The average bits per dimension metric of the batch.
                  This is also the loss we train on. Shape: single scalar
        """
        
        imgs = imgs.to(self.device)
        
        mean, log_std = self.encoder(imgs)
        z = sample_reparameterize(mean, log_std)
        x_hat = torch.sigmoid(self.decoder(z))

        elbo, L_rec, L_reg = ELBO(imgs, x_hat, mean, log_std)

        C = self.information_flow()

        bpd = ELBO_to_BPD(elbo, imgs.size())
        
        return torch.mean(L_rec), torch.mean(L_reg), C, torch.mean(bpd)
    
    def information_flow(self):
        # Computes an MC-estimator for the information flow
        # This is a literal implementation of Algorithm 2, Appendix D
        
        #C = 0.0
        #q_y = torch.zeros((1, self.M))

        #for i in range(self.Nalpha):
        #    alpha = torch.normal(0, 1, size=(1, self.K))
        #    p_y_a = torch.zeros((1, self.M))
        #    for j in range(self.Nbeta):
        #        beta = torch.normal(0, 1, size=(1, self.L))
        #        x = self.decoder(torch.cat((alpha, beta), dim=1))

        #        p_y_a += F.softmax(self.classifier(x), dim=1)/self.Nbeta

        #    C += torch.sum(p_y_a * torch.log(p_y_a))/self.Nalpha
        #    q_y += p_y_a/self.Nalpha

        #C -= torch.sum(q_y * torch.log(q_y))/self.Nalpha
        
        C, debug = joint_uncond(*CVAE_to_params(self))
        
        return C

    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr,
                                     betas=(0.5, 0.999)
                                     )
        return optimizer

    def training_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, C, bpd = self.forward(batch[0])
        self.log("Train Reconstruction Loss", L_rec, 
                 on_step=True, on_epoch=False)
        self.log("Train Regularization Loss", L_reg, 
                 on_step=True, on_epoch=False)
        self.log("Train Information Flow", C, on_step=True, on_epoch=False)
        
        self.log("Train ELBO", L_rec + L_reg, on_step=True, on_epoch=False)
        self.log("Train Causal Loss", C + self.lamb*(L_rec + L_reg), on_step=True, on_epoch=False)
        self.log("Train BPD", bpd, on_step=True, on_epoch=False)
        
        return self.use_ce * C + self.lamb*(L_rec + L_reg) #bpd

    def validation_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, C, bpd = self.forward(batch[0])
        self.log("Valid Reconstruction Loss", L_rec,
                 on_step=False, on_epoch=True)
        self.log("Valid Regularization Loss", L_reg,
                 on_step=False, on_epoch=True)
        self.log("Valid Information Flow", C, on_step=False, on_epoch=True)

        self.log("Valid ELBO", L_rec + L_reg, on_step=False, on_epoch=True)
        self.log("Valid Causal Loss", C + self.lamb*(L_rec + L_reg), on_step=False, on_epoch=True)
        self.log("Valid BPD", bpd, on_step=False, on_epoch=True)
        
        # print("Causal:", C, "ELBO:", (L_rec + L_reg).item(), "BPD:", bpd.item())

    def test_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        L_rec, L_reg, C, bpd = self.forward(batch[0])
        self.log("Test Information Flow", C, on_step=False, on_epoch=True)
        self.log("Test ELBO", L_rec + L_reg, on_step=False, on_epoch=True)
        self.log("Test BPD", bpd, on_step=False, on_epoch=True)
        self.log("Test Causal Loss", C + self.lamb*(L_rec + L_reg), on_step=False, on_epoch=True)        
        
    @torch.no_grad()
    def sample(self, batch_size: int):
        """
        Function for sampling a new batch of random images.
        Inputs:
            batch_size - Number of images to generate
        Outputs:
            x_samples - Sampled, binarized images with 0s and 1s. Shape: [B,C,H,W]
            x_mean - The sigmoid output of the decoder with continuous values
                     between 0 and 1 from which we obtain "x_samples".
                     Shape: [B,C,H,W]
        """
        z = torch.normal(0, 1, size=(batch_size, self.z_dim),
                         device=self.device)
        x_mean = torch.sigmoid(self.decoder(z)).detach()
        x_samples = torch.round(x_mean)

        return x_samples, x_mean
    
    @torch.no_grad()
    def sample(self, z: torch.Tensor):
        """
        Function for sampling a new batch of random images.
        Inputs:
            z - The latent variables, consisting of both alpha and beta
        Outputs:
            x_samples - Sampled, binarized images with 0s and 1s. Shape: [B,C,H,W]
            x_mean - The sigmoid output of the decoder with continuous values
                     between 0 and 1 from which we obtain "x_samples".
                     Shape: [B,C,H,W]
        """
        
        x_mean = torch.sigmoid(self.decoder(z)).detach()
        x_samples = torch.round(x_mean)

        return x_samples, x_mean


class GenerateCallback(pl.Callback):

    def __init__(self, batch_size=64, every_n_epochs=1, save_to_disk=True, valid_data=None):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.batch_size = batch_size
        self.every_n_epochs = every_n_epochs
        self.save_to_disk = save_to_disk
        
        self.valid_data = valid_data


    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if ((trainer.current_epoch+1) % self.every_n_epochs == 0 or
            trainer.current_epoch == 0 or
                (trainer.current_epoch+1) == trainer.max_epochs):
            #self.sample_and_save(trainer, pl_module, trainer.current_epoch+1)
            self.sweep_and_save(trainer, pl_module, trainer.current_epoch+1)

    def sample_and_save(self, trainer, pl_module, epoch):
        """
        Function that generates and save samples from the VAE.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
            epoch - The epoch number to use for TensorBoard logging and saving of the files.
        """

        imgs, _ = pl_module.sample(64)

        if self.save_to_disk:
            save_image(imgs, trainer.logger.log_dir+'/epoch{:d}.png'.format(epoch),
                       nrow=8)

        img_grid = make_grid(imgs, nrow=8)
        img_grid = img_grid.mul(255).add_(
            0.5).clamp_(0, 255)  # .permute(1, 2, 0)
        img_grid = img_grid.type(torch.ByteTensor).numpy()

        trainer.logger.experiment.add_image('Generated Digits',
                                            img_grid, epoch)

        return img_grid
    
    def sweep_and_save(self, trainer, pl_module, epoch):
        """
        Function that sweeps over all latent variables and saves samples from the VAE.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
        """

        img_grid = []
        for i in range(pl_module.K+pl_module.L):
            img_grid.append(
                CVAE_sweep(pl_module, i=i, rows=8,
                           dataset=self.valid_data, save_loc=epoch)
            )

        return img_grid


def train_vae(args):
    """
    Function for training and testing a VAE model.
    Inputs:
        args - Namespace object from the argument parser
    """
    
    torch.autograd.set_detect_anomaly = True
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Handling the training
    train_set, valid_set = MNIST_limited(train=True)
    test_set = MNIST_limited(train=False)

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                   drop_last=True, pin_memory=True, num_workers=0)
    valid_loader = data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                                   drop_last=True, pin_memory=True, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                  drop_last=True, pin_memory=True, num_workers=0)

    # Create a PyTorch Lightning trainer with the generation callback
    gen_callback = GenerateCallback(save_to_disk=True, every_n_epochs=1, valid_data=valid_set)
    
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=ModelCheckpoint(
                             save_weights_only=True, mode="min", monitor="Train Causal Loss"),
                         gpus= 1 if (torch.cuda.is_available() and args.gpu) else 0,
                         max_epochs=args.max_epochs, 
                         val_check_interval=1.0,
                         callbacks=[gen_callback],
                         progress_bar_refresh_rate=5 if args.progress_bar else 0,
                         fast_dev_run=args.fast_dev_run
                         )
    
    # Optional logging argument that we don't need
    trainer.logger._default_hp_metric = None
    #trainer.logger._version = str(args.model) + '_' + str(args.z_dim) + '_' + str(args.seed)

    # Create model
    pl.seed_everything(args.seed)  # To be reproducible
    model = MNIST_VAE(args.classifier_name, 
                      num_filters=args.num_filters, 
                      K=args.K, L=args.L, M=args.M,
                      lamb=args.lamb, lr=args.lr, 
                      Nalpha=args.Nalpha, Nbeta=args.Nbeta)

    # Training
    trainer.fit(model, train_loader, valid_loader)

    # Testing
    model = load_latest(MNIST_VAE, 'MNIST_VAE')
    test_result = trainer.test(
        model, test_dataloaders=test_loader, verbose=True)

    return test_result, trainer


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Model hyperparameters
    parser.add_argument('--classifier_name', default='MNIST_CNN_OShaugnessy', 
                        type=str, help='The black-boz classifier we wish to explain.')
    parser.add_argument('--num_filters', default=64, type=int,
                        help='Number of filters used in the encoders/decoders')
    parser.add_argument('--K', default=1, type=int,
                        help='Dimensionality of causal latent space')
    parser.add_argument('--L', default=7, type=int,
                        help='Dimensionality of non-causal latent space')
    parser.add_argument('--M', default=2, type=int,
                        help='Dimensionality of classifier output')
    parser.add_argument('--lamb', default=0.05, type=float,
                        help='VAE-loss coefficient')

    # Loss and optimizer hyperparameters
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Learning rate to use')
    parser.add_argument('--Nalpha', default=32, type=int,
                        help='Learning rate to use')
    parser.add_argument('--Nbeta', default=16, type=int,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--max_epochs', default=20, type=int,
                        help='Max number of training epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0. ' +
                             'For your assignment report, you can use multiple workers (e.g. 4) and do not have to set it to 0.')
    parser.add_argument('--log_dir', default='Models/MNIST_VAE', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', default=True, action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))
    
    # Debug parameters
    parser.add_argument('--fast_dev_run', default=False, 
                        help=('Whether to check debugs, etc.'))
    parser.add_argument('--gpu', default=True, action='store_true',
                        help=('Whether to train on GPU (if available) or CPU'))

    args = parser.parse_args()

    test_result, trainer = train_vae(args)
