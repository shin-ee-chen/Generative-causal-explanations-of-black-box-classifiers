import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchtext
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models.lm_cvae import text_VAE
from datasets.sst import SST, get_glove_url
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
            #self.sweep_and_save(trainer, pl_module, trainer.current_epoch)

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

        text_sample = pl_module.reconstruct(batch.text)
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

    if args.version == '':
        full_log_dir = os.path.join(CHECKPOINT_PATH, args.log_dir)
    else:
        full_log_dir = os.path.join(CHECKPOINT_PATH, args.log_dir + '_' + args.version)
    print('Saving to:', full_log_dir)
    os.makedirs(full_log_dir, exist_ok = True)

    # Handling the training
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')

    data_loaders, info = SST.iters(batch_size=args.batch_size, repeat=True,
                                   fine_grained=False, device=device)
    (train_loader, valid_loader, test_loader) = data_loaders
    (vocab, train_data) = info

    gen_callback = GenerateCallback(every_n_epochs=args.sample_every,
                                    data_loader=valid_loader,
                                    save_loc=full_log_dir)

    checkpoint_callback = ModelCheckpoint(monitor='Valid ELBO',
                                          filename='text_vae-epoch{epoch:02d}-val_elbo{Valid ELBO:.2f}-val_kld{Valid L_reg:.2f}--val_mi{Valid MI:.2f}',
                                          mode='min',
                                          verbose=True,
                                          save_last=True)

    trainer = pl.Trainer(default_root_dir=full_log_dir,
                         gpus=1 if (torch.cuda.is_available() and args.gpu) else 0,
                         max_epochs=args.max_epochs,
                         callbacks=[checkpoint_callback, gen_callback],
                         progress_bar_refresh_rate=1 if args.progress_bar else 0,
                         fast_dev_run=args.debug,
                         gradient_clip_val=5
                        )

    trainer.logger._default_hp_metric = None

    if args.debug:
        trainer.logger._version = 'debug'  # str(args.model) + '_' + str(args.z_dim) + '_' + str(args.seed)
    elif args.version != '':
        trainer.logger._version = args.version

    set_seed(42)
    #set_deterministic()

    anneal_rate = (1.0 - args.kl_weight_start) / (args.warm_up * (len(train_data) / args.batch_size))
    model = text_VAE(vocab=vocab,
                     latent_dims=args.latent_dims,
                     n_layers=args.n_layers,
                     embedding_dims = args.embedding_dims,
                     hidden_dims=args.hidden_dims,
                     dropout=args.drop_out,
                     teacher_force_p=args.teacher_force_p,
                     lr=args.lr,
                     decoding_strategy=args.decoding_strategy,
                     aggressive=args.aggressive,
                     inner_iter=args.inner_iter,
                     kl_weight_start=args.kl_weight_start,
                     anneal_rate=anneal_rate)
    model.automatic_optimization = False

    trainer.fit(model, train_loader, valid_loader)

    # Eval post training
    model = text_VAE.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)

    test_result = trainer.test(
        model, test_dataloaders=test_loader, verbose=True)

    return test_result, trainer

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--latent_dims', default=32, type=int,
                        help='Number of latent variables to encode to')
    parser.add_argument('--n_layers', default=1, type=int,
                        help='Number of LSTM layers')
    parser.add_argument('--embedding_dims', default=512, type=int,
                        help='Dimesionality of embedding space.')
    parser.add_argument('--hidden_dims', default=1024, type=int,
                        help='Number of LSTM hidden neurons. \
                            For this model, also the latent dimensions.')
    parser.add_argument('--drop_out', default=0.5, type=float,
                        help='Probability of zeroing a neuron.')
    parser.add_argument('--teacher_force_p', default=0.5, type=float,
                        help='Probability of feeding correct answer. \
                            Stabilizes training.')
    parser.add_argument('--decoding_strategy', default='greedy', type=str,
                        choices=['greedy', 'sample'],
                        help='Strategy for decoding, whether sampling \
                            or reconstructing.')

    # Loss and optimizer hyperparameters
    parser.add_argument('--lr', default=1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size')
    parser.add_argument('--max_epochs', default=100, type=int,
                        help='Max number of training batches')

    # Aggressive Training hyperparameters
    parser.add_argument('--aggressive', default=True,
                        help='Whether or not to use aggressive training')
    parser.add_argument('--inner_iter', default=100, type=int,
                        help='Number of steps before assuming encoder has converged')
    parser.add_argument('--kl_weight_start', default=0.1, type=float,
                        help='Weight of the KLD term in the first step of training')
    parser.add_argument('--warm_up', default=10, type=int,
                        help='Number of epochs to warmup KLD weight')

    # Other hyperparameters
    parser.add_argument('--fine_grained', default=False,
                        help='Whether to train using 2 or 5 sentiment classes')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--progress_bar', default=False, action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))
    parser.add_argument('--sample_every', default=1, type=int,
                        help='When to sample the latent space. If -1, only samples at end of training.')
    parser.add_argument('--log_dir', default='text_vae', type=str,
                        help='Directory where the PyTorch Lightning logs should be created. Automatically adds \
                            the classes to directory. If not needed, turn off using add_classes_to_cpt_path flag.')
    parser.add_argument('--version', default='Test', type=str,
                        help='Run name. For example, SLURM jobid.')

    # Debug parameters
    parser.add_argument('--debug', default=False,
                        help=('Whether to check debugs, etc.'))
    parser.add_argument('--gpu', default=True, action='store_true',
                        help=('Whether to train on GPU (if available) or CPU'))

    args = parser.parse_args()

    test_result, trainer = train(args)
