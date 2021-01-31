import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from models.lm_gce import lm_gce
from models.lm_vae import lm_VAE
from models.sst_bilstm_cnn import sst_bilstm_cnn
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
              trainer.current_epoch == trainer.max_steps):

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

        for i in range(16):
            text_sample = pl_module.latent_sweep(batch.text[i][None, :], zi=0)
            path = os.path.join(self.save_loc,
                                'z_0({:d}).txt'.format(i))
            np.savetxt(path, text_sample,
                       fmt='%s: %s',
                       header='Sweep over z_0({:d})'.format(i))

        for zi in range(1, pl_module.vae.latent_dims):
            text_sample = pl_module.latent_sweep(batch.text, zi=zi)
            path = os.path.join(self.save_loc,
                                'z_{:d}.txt'.format(zi))
            np.savetxt(path, text_sample,
                       fmt='%s: %s',
                       header='Sweep over z_%d'.format(zi))

def train(args):

    full_log_dir = os.path.join(CHECKPOINT_PATH, args.log_dir + ('_5' if args.fine_grained else '_2'))
    os.makedirs(full_log_dir, exist_ok=True)
    print('Saving models to', full_log_dir)

    # Handling the training
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')
    data_loaders, info = SST.iters(batch_size=args.batch_size, repeat=True,
                                   finegrained=args.fine_grained, device=device,
                                   pad_to_max=True)
    (train_loader, valid_loader, test_loader) = data_loaders
    (vocab, train_data) = info

    gen_callback = GenerateCallback(every_n_epochs=args.sample_every,
                                    data_loader=train_loader,
                                    save_loc=full_log_dir)

    checkpoint_callback = ModelCheckpoint(mode="min",
                                          monitor="Valid Causal Loss",
                                          save_last=True,
                                          save_top_k=3,
                                          verbose=True)

    trainer = pl.Trainer(default_root_dir=full_log_dir,
                         checkpoint_callback=checkpoint_callback,
                         gpus=1 if (torch.cuda.is_available() and args.gpu) else 0,
                         max_epochs=args.max_epochs,
                         callbacks=[gen_callback],
                         progress_bar_refresh_rate=1 if args.progress_bar else 0,
                         fast_dev_run=args.debug,
                         gradient_clip_val=5
                        )

    trainer.logger._default_hp_metric = None

    set_seed(42)
    set_deterministic()

    if args.debug:
        trainer.logger._version = 'debug'
    elif args.version != '':
        trainer.logger._version = args.version

    gce = lm_gce(clf_path=args.classifier_path,
                 vae_path=args.vae_path,
                 K=args.K,
                 lamb=args.lamb,
                 Nalpha=args.Nalpha,
                 Nbeta=args.Nbeta,
                 lr=args.lr,
                 betas=tuple(args.betas))

    trainer.fit(gce, train_loader, valid_loader)

    print('\nlr:', args.lr, 'lamb:', args.lamb)

    # Eval post training
    model = lm_gce.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)

    test_result = trainer.test(
        model, test_dataloaders=test_loader, verbose=True)

    return test_result, trainer

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--fine_grained', default=False,
                        help='Whether to train using 2 or 5 sentiment classes')
    parser.add_argument('--classifier_path', default='', type=str,
                        help='This is the directory INSIDE of models where the pre-trained \
                            black-box classifier is. Necessary if naming convention is not \
                            adhered to')
    parser.add_argument('--vae_path', default='', type=str,
                        help='This is the directory INSIDE of models where the pre-trained \
                            generative VAE is. Necessary if naming convention is not \
                            adhered to')
    parser.add_argument('--lamb', default=0.001, type=float,
                        help='VAE-loss coefficient')
    parser.add_argument('--K', default=1, type=int,
                        help='Dimensionality of causal latent space')

    # Loss and optimizer hyperparameters
    parser.add_argument('--max_epochs', default=20, type=int,
                        help='Max number of training batches')
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--Nalpha', default=250, type=int,
                        help='Learning rate to use')
    parser.add_argument('--Nbeta', default=10, type=int,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size')
    parser.add_argument('--betas', default=[0.5, 0.99],
                        type=int, nargs=2,
                        help='The beta parameters for add_argument')

    # Other hyperparameters
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--progress_bar', default=False, action='store_true',
                        help='Use a progress bar indicator for interactive experimentation.\
                              Not to be used in conjuction with SLURM jobs')
    parser.add_argument('--sample_every', default=1, type=int,
                        help='When to sample the latent space. If -1, only samples at end of training.')
    parser.add_argument('--log_dir', default='sst_lm_gce', type=str,
                        help='Directory where the PyTorch Lightning logs should be created. Automatically adds \
                            the classes to directory. If not needed, turn off using add_classes_to_cpt_path flag.')
    parser.add_argument('--version', default='', type=str,
                        help='Version name.')

    # Debug parameters
    parser.add_argument('--debug', default=False,
                        help=('Whether to check debugs, etc.'))
    parser.add_argument('--gpu', default=True, action='store_true',
                        help=('Whether to train on GPU (if available) or CPU'))

    args = parser.parse_args()

    test_result, trainer = train(args)
