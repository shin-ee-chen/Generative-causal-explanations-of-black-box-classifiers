import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from distutils.util import strtobool

from models.cvae import MNIST_CVAE
from datasets.mnist import MNIST_limited
from datasets.fashion_mnist import Fashion_MNIST_limited

from utils.cvae_latent_visualization import CVAE_sweep
from utils.reproducibility import set_seed, set_deteministic, load_latest
from utils.timing import Timer

CHECKPOINT_PATH = './checkpoints'

class GenerateCallback(pl.Callback):

    def __init__(self, batch_size, every_n_epochs, save_to_disk, save_dir, valid_data=None):
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
        self.save_dir = save_dir

        self.valid_data = valid_data

    def on_fit_end(self, trainer, pl_module):
        """
        This function is called after finishing training.
        """
        if self.every_n_epochs == -1:
            self.sweep_and_save(trainer, pl_module, save_loc=self.save_dir)

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        """
        if self.every_n_epochs == -1:
            pass

        elif ((trainer.current_epoch + 1) % self.every_n_epochs == 0 or
            trainer.current_epoch == 0 or
                (trainer.current_epoch + 1) == trainer.max_epochs):
            #self.sample_and_save(trainer, pl_module, trainer.current_epoch+1)
            self.sweep_and_save(trainer, pl_module, save_loc=os.path.join(self.save_dir, trainer.current_epoch))
            
        torch.cuda.empty_cache()

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
            save_image(imgs, os.path.join(self.save_dir, '/epoch{:d}.png'.format(epoch)),
                       nrow=8)

        img_grid = make_grid(imgs, nrow=8)
        img_grid = img_grid.mul(255).add_(
            0.5).clamp_(0, 255)  # .permute(1, 2, 0)
        img_grid = img_grid.type(torch.ByteTensor).numpy()

        trainer.logger.experiment.add_image('Generated Digits',
                                            img_grid, epoch)

        return img_grid

    def sweep_and_save(self, trainer, pl_module, save_loc):
        """
        Function that sweeps over all latent variables and saves samples from the VAE.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
        """

        img_grid = []
        for i in range(pl_module.K + pl_module.L):
            img_grid.append(
                CVAE_sweep(pl_module, i=i, rows=self.batch_size,
                           dataset=self.valid_data, save_loc=save_loc)
            )

def train(args):
    """
    Function for training and testing a VAE model.
    Inputs:
        args - Namespace object from the argument parser
    """

    if args.add_classes_to_log_dir == True:
        classes_str = ''.join(str(x) for x in sorted(args.classes))
        full_log_dir = os.path.join(CHECKPOINT_PATH, args.log_dir + '_' + classes_str)
    else:
        full_log_dir = os.path.join(CHECKPOINT_PATH, args.log_dir)
    os.makedirs(full_log_dir, exist_ok=True)
    os.makedirs(os.path.join(full_log_dir, "lightning_logs"), exist_ok=True) # to fix "Missing logger folder"

    # Handling the training
    # train_set, valid_set = MNIST_limited(train=True, classes=args.classes)
    # test_set = MNIST_limited(train=False, classes=args.classes)
    if args.datasets == 'traditional':
        train_set, valid_set = MNIST_limited(train=True, classes=args.classes)
        test_set = MNIST_limited(train=False, classes=args.classes)
    else:
        train_set, valid_set = Fashion_MNIST_limited(train=True, classes=args.classes)
        test_set = Fashion_MNIST_limited(train=False, classes=args.classes)
        print(f"train_set:{len(train_set)}")

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                   drop_last=True, pin_memory=True, num_workers=args.num_workers)
    valid_loader = data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                                   drop_last=True, pin_memory=True, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                  drop_last=True, pin_memory=True, num_workers=args.num_workers)
    
    if args.silent:
        callbacks = []
        import logging
        logging.getLogger('lightning').setLevel(logging.WARNING)
    else:
        callbacks = [GenerateCallback(batch_size=8, save_to_disk=True, save_dir = args.log_dir, every_n_epochs=args.sample_every, valid_data=valid_set)]
    
    set_deteministic()

    set_seed(42)

    trainer = pl.Trainer(default_root_dir=full_log_dir,
                         checkpoint_callback=ModelCheckpoint(
                             save_weights_only=True, mode="min", monitor="Valid Causal Loss"),
                         gpus=1 if (torch.cuda.is_available() and args.gpu) else 0,
                         max_steps=args.max_steps,
                         val_check_interval=1.0,
                         callbacks=callbacks,
                         progress_bar_refresh_rate=5 if args.progress_bar and not args.silent else 0,
                         weights_summary=None if args.silent else 'top',
                         fast_dev_run=args.debug
                         )
    
    trainer.logger._default_hp_metric = None
    
    if args.debug:
        trainer.logger._version =  'debug' # str(args.model) + '_' + str(args.z_dim) + '_' + str(args.seed)
    
    model = MNIST_CVAE(args.classes,
                      num_filters=args.num_filters,
                      K=args.K, L=args.L, M=len(args.classes),
                      lamb=args.lamb, lr=args.lr,
                      betas=args.betas,
                      Nalpha=args.Nalpha, Nbeta=args.Nbeta,
                      classifier_path=args.classifier_path,
                      use_C = args.use_C,
                      silent = args.silent)
    
    timer = Timer(args.silent)
    trainer.fit(model, train_loader, valid_loader)
    if not args.silent: print(f"Total training time: {timer.time()}")

    # Eval post training
    model = MNIST_CVAE.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)

    test_result = trainer.test(
        model, test_dataloaders=test_loader, verbose=not args.silent)
    
    gce_path = os.path.join('pretrained_models', args.log_dir)
    if not os.path.exists(gce_path):
        os.makedirs(gce_path, exist_ok=True)

    torch.save(model, os.path.join(gce_path,'cvae_model.pt'))
    
    return test_result, trainer





if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--classes', default=[3, 8],
                        type=int, nargs='+',
                        help='The classes permittible for classification')
    parser.add_argument('--classifier_path', type=str,
                        help='This is the directory INSIDE of models where pre-trained \
                            black-box classifier is. Necessary if naming convention is not \
                                adhered to')
    parser.add_argument('--num_filters', default=64, type=int,
                        help='Number of filters used in the encoders/decoders')
    parser.add_argument('--K', default=1, type=int,
                        help='Dimensionality of causal latent space')
    parser.add_argument('--L', default=7, type=int,
                        help='Dimensionality of non-causal latent space')
    parser.add_argument('--lamb', default=0.05, type=float,
                        help='VAE-loss coefficient')
    parser.add_argument('--use_C', default=True, type=lambda x: bool(strtobool(x)),
                        help='Whether or not the causal influence term should be optimized along with the VAE loss.')

    # Loss and optimizer hyperparameters
    parser.add_argument('--max_steps', default=8000, type=int,
                        help='Max number of training batches')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Learning rate to use')
    parser.add_argument('--Nalpha', default=100, type=int,
                        help='Learning rate to use')
    parser.add_argument('--Nbeta', default=25, type=int,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size')
    parser.add_argument('--betas', default=[0.5, 0.99],
                        type=int, nargs=2,
                        help='The beta parameters for add_argument')

    # Other hyperparameters
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--progress_bar', default=True, type=lambda x: bool(strtobool(x)),
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))
    parser.add_argument('--sample_every', default=-1, type=int,
                        help='When to sample the latent space. If -1, only samples at end of training.')
    parser.add_argument('--log_dir', default='mnist_cvae', type=str,
                        help='Directory where the PyTorch Lightning logs should be created. Automatically adds \
                            the classes to directory. If not needed, turn off using add_classes_to_cpt_path flag.')
    parser.add_argument('--add_classes_to_log_dir', default=True, type=lambda x: bool(strtobool(x)),
                        help='Whether to add the classes to log directory name.')
    parser.add_argument('--silent', default=False, type=lambda x: bool(strtobool(x)),
                        help='Perform training without printing to console or creating graphs.')
    parser.add_argument('--datasets', default='traditional',choices=['traditional', 'fashion'],
                        help='Datasets used for training: traditional or fashion')

    # Debug parameters
    parser.add_argument('--debug', default=False, type=lambda x: bool(strtobool(x)),
                        help=('Whether to check debugs, etc.'))
    parser.add_argument('--gpu', default=True, type=lambda x: bool(strtobool(x)),
                        help=('Whether to train on GPU (if available) or CPU'))
    parser.add_argument('--num_workers', default=0, type=int,
                        help=('Number of workers to use for the dataloaders.'))

    args = parser.parse_args()

    test_result, trainer = train(args)
