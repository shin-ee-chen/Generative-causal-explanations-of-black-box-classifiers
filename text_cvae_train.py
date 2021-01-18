import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models.lm_cvae import lm_VAE
from datasets.sst import SST
from utils.reproducibility import set_seed, set_deterministic, load_latest

CHECKPOINT_PATH = './checkpoints'

class GenerateCallback(pl.Callback):

    def __init__(self, save_loc, every_n_epochs, data_loader):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
        self.save_loc = save_loc
        self.every_n_epochs = every_n_epochs
        self.data_loader = iter(data_loader)

    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        """
        if self.every_n_epochs == -1:
            pass

        elif ((trainer.current_epoch + 1) % self.every_n_epochs == 0 or
              trainer.current_epoch == 0 or
              trainer.current_epoch == trainer.max_epochs):

            self.save_loc = os.path.join(trainer.logger.log_dir,
                                         'epoch{: d}'.format(trainer.current_epoch))
            os.makedirs(self.save_loc,
                        exist_ok=True)

            self.sample_and_save(trainer, pl_module, trainer.current_epoch)
            self.sweep_and_save(trainer, pl_module, trainer.current_epoch)

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

        try:
            batch = next(self.data_loader)
        except StopIteration:
            self.data_loader = iter(self.data_loader)
            batch = next(self.data_loader)

        text_sample = pl_module.decode(batch.text)
        path = os.path.join(self.save_loc,
                            'samples.txt')
        np.savetxt(path, text_sample,
                fmt='%s',
                header='Reconstructed Samples')

    def sweep_and_save(self, trainer, pl_module, epoch):
        """
        Function that sweeps over all latent variables and saves samples from the VAE.
        The generated samples and mean images should be added to TensorBoard and,
        if self.save_to_disk is True, saved inside the logging directory.
        Inputs:
            trainer - The PyTorch Lightning "Trainer" object.
            pl_module - The VAE model that is currently being trained.
        """

        try:
            batch = next(self.data_loader)
        except StopIteration:
            self.data_loader = iter(self.data_loader)
            batch = next(self.data_loader)

        for zi in range(pl_module.latent_dims):
            text_sample = pl_module.latent_sweep(batch.text, zi=zi)
            path = os.path.join(self.save_loc,
                                'z_{:d}.txt'.format(zi))
            np.savetxt(path, text_sample,
                    fmt='%s: %s',
                    header='Sweep over z_%d' % (zi))

def train(args):
    """
    Function for training and testing a VAE model.
    Inputs:
        args - Namespace object from the argument parser
    """

    #assert len(args.classes) == args.M

    #if args.add_classes_to_cpt_path == True:
    #    classes_str = ''.join(str(x) for x in sorted(args.classes))
    #    full_log_dir = os.path.join(CHECKPOINT_PATH, args.log_dir + '_' + classes_str)
    #else:
    full_log_dir = os.path.join(CHECKPOINT_PATH, args.log_dir)
    os.makedirs(full_log_dir, exist_ok = True)

    # Handling the training
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')
    data_loaders, info = SST.iters(batch_size=args.batch_size, repeat=True, device=device)
    (train_loader, valid_loader, test_loader) = data_loaders
    (vocab, train_data) = info

    gen_callback = GenerateCallback(every_n_epochs=args.sample_every,
                                    data_loader=valid_loader,
                                    save_loc=full_log_dir)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(default_root_dir=full_log_dir,
                         checkpoint_callback=ModelCheckpoint(
                             save_weights_only=True, mode="min", monitor="Valid BPD"),
                         gpus=1 if (torch.cuda.is_available() and args.gpu) else 0,
                         max_epochs=args.max_epochs,
                         callbacks=[gen_callback, lr_monitor],
                         progress_bar_refresh_rate=1 if args.progress_bar else 0,
                         fast_dev_run=args.debug,
                         gradient_clip_val=5,
                         automatic_optimization=False
                         )

    trainer.logger._default_hp_metric = None

    set_seed(42)
    set_deterministic()

    if args.debug:
        trainer.logger._version = 'debug'  # str(args.model) + '_' + str(args.z_dim) + '_' + str(args.seed)

    anneal_rate = (1.0 - args.kl_start) / (args.warm_up * (len(train_data) / args.batch_size))
    model = lm_VAE(vocab=vocab,
                   embedding_dims=512,
                   hidden_dims=1024,
                   latent_dims=32,
                   z_iters=50,
                   aggressive=True,
                   inner_iter=30,
                   kl_weight_start=args.kl_start,
                   anneal_rate=anneal_rate,
                   decoding_strategy='beam_search')

    trainer.fit(model, train_loader, valid_loader)

    # Eval post training
    model = lm_VAE.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)

    test_result = trainer.test(
        model, test_dataloaders=test_loader, verbose=True)

    return test_result, trainer


if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--kl_start', default=0.1, type=float,
                        help='Start weight of KLD')
    parser.add_argument('--warm_up', default=10, type=int,
                        help='Number of epochs of KLD weight warm-up')

    # Loss and optimizer hyperparameters
    parser.add_argument('--max_epochs', default=10, type=int,
                        help='Max number of training batches')
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--progress_bar', default=True, action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))
    parser.add_argument('--sample_every', default=1, type=int,
                        help='When to sample the latent space. If -1, only samples at end of training.')
    parser.add_argument('--log_dir', default='text_vae', type=str,
                        help='Directory where the PyTorch Lightning logs should be created. Automatically adds \
                            the classes to directory. If not needed, turn off using add_classes_to_cpt_path flag.')

    # Debug parameters
    parser.add_argument('--debug', default=False,
                        help=('Whether to check debugs, etc.'))
    parser.add_argument('--gpu', default=True, action='store_true',
                        help=('Whether to train on GPU (if available) or CPU'))

    args = parser.parse_args()

    test_result, trainer = train(args)
