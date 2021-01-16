import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.utils.data as data

from models.mnist_cnn import MNIST_CNN
from utils.reproducibility import set_seed, set_deteministic
from datasets.mnist import MNIST_limited

from models.cvae import MNIST_CVAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib.pyplot as plt

class GenerateCallback(pl.Callback):

    def __init__(self, batch_size, every_n_epochs, save_to_disk, valid_data=None):
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

    def on_fit_end(self, trainer, pl_module):
        """
        This function is called after finishing training.
        """
        if self.every_n_epochs == -1:
            self.sweep_and_save(trainer, pl_module, save_loc=trainer.logger._version)

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
            self.sweep_and_save(trainer, pl_module, save_loc=trainer.logger._version+'_'+trainer.current_epoch)

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
            save_image(imgs, trainer.logger.log_dir + '/epoch{:d}.png'.format(epoch),
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
    Inputs:
        args - Namespace object from the argparser
    """

    M = len(args.classes)

    train_set, valid_set = MNIST_limited(train=True, classes=args.classes)
    test_set = MNIST_limited(train=False, classes=args.classes)

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                   drop_last=True, pin_memory=True, num_workers=0)
    valid_loader = data.DataLoader(valid_set, batch_size=args.batch_size, shuffle=False,
                                   drop_last=True, pin_memory=True, num_workers=0)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                  drop_last=True, pin_memory=True, num_workers=0)



    # load classifier
    classifier = MNIST_CNN(model_param_set=args.clf_param_set, M=M,
                        lr=args.lr, momentum=args.momentum)

    classifier_path = './pretrained_models/mnist_cnn/'
    checkpoint_model = torch.load(os.path.join(classifier_path,'model.pt'), map_location=device)
    classifier.load_state_dict(checkpoint_model['model_state_dict_classifier'])

    # load GCE
    gce_path = './pretrained_models/mnist_cvae/'
    gce = torch.load(os.path.join(gce_path,'model.pt'), map_location=device)

    # plot information_flow
    z_dim = args.K + args.L
    info_flow = gce.information_flow_single(range(0,z_dim))

    cols = {'golden_poppy' : [1.000,0.761,0.039],
        'bright_navy_blue' : [0.047,0.482,0.863],
        'rosso_corsa' : [0.816,0.000,0.000]}
    x_labels = ('$\\alpha_1$', '$\\alpha_2$', '$\\beta_1$', '$\\beta_2$')
    fig, ax = plt.subplots()
    ax.bar(range(z_dim), info_flow, color=[
        cols['rosso_corsa'], cols['rosso_corsa'], cols['bright_navy_blue'],
        cols['bright_navy_blue']])
    plt.xticks(range(z_dim), x_labels)
    ax.yaxis.grid(linewidth='0.3')
    plt.ylabel('Information flow to $\\widehat{Y}$')
    plt.title('Information flow of individual causal factors')
    plt.savefig('./figures/fig5a.svg')
    plt.savefig('./figures/fig5a.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--clf_param_set', default='OShaugnessy',
                        type=str, help='The black-box classifier we wish to explain.')
    parser.add_argument('--classes', default=[3, 8],
                        type=int, nargs='+',
                        help='The classes permittible for classification')

    # Loss and optimizer hyperparameters
    parser.add_argument('--lr', default=5e-4, type=float,
                        help='Learning rate to use')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size')
    parser.add_argument('--max_epochs', default=1, type=int,
                        help='Max number of training epochs')

    # Other hyperparameters

    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders.')
    parser.add_argument('--progress_bar', default=True, action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))
    parser.add_argument('--log_dir', default='mnist_cnn', type=str,
                        help='Directory where the PyTorch Lightning logs should be created. Automatically adds \
                            the classes to directory. If not needed, turn off using add_classes_to_cpt_path flag.')
    parser.add_argument('--add_classes_to_cpt_path', default=True,
                        help='Whether to add the classes to cpt directory.')


    # Debug parameters
    parser.add_argument('--debug_version', default=False,
                        help=('Whether to check debugs, etc.'))
    parser.add_argument('--fast_dev_run', default=False,
                        help=('Whether to check debugs, etc.'))
    parser.add_argument('--gpu', default=True, action='store_true',
                        help=('Whether to train on GPU (if available) or CPU'))

    # param for cvae
    parser.add_argument('--K', default=2, type=int,
                       help='Dimensionality of causal latent space')
    parser.add_argument('--L', default=2, type=int,
                       help='Dimensionality of non-causal latent space')
    parser.add_argument('--M', default=3, type=int,
                       help='Dimensionality of classifier output')

    args = parser.parse_args()

    train(args)
