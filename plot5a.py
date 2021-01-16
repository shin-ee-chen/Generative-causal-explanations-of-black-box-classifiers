import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch.utils.data as data

from models.mnist_cnn import MNIST_CNN
from utils.reproducibility import set_seed, set_deteministic
from datasets.mnist import MNIST_limited

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

    model = MNIST_CNN(model_param_set=args.clf_param_set, M=M,
                        lr=args.lr, momentum=args.momentum)
    trainer.fit(model, train_loader, valid_loader)

    # Eval post training
    classifier = MNIST_CNN.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path)

    classifier_path = './pretrained_models/mnist_cnn/'
    checkpoint_model = torch.load(os.path.join(classifier_path,'model.pt'), map_location=device)
    classifier.load_state_dict(checkpoint_model['model_state_dict_classifier'])

    # Test results
    val_result = trainer.test(
        model, test_dataloaders=valid_loader, verbose=False)
    test_result = trainer.test(
        model, test_dataloaders=test_loader, verbose=False)
    result = {"Test": test_result[0]["Test_acc"],
              "Valid": val_result[0]["Test_acc"]}
    save_folder = './pretrained_models/'+ args.log_dir + '/'

    torch.save({
    'model_state_dict_classifier': model.state_dict()
        }, os.path.join(save_folder, 'model.pt'))

    return model, result

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

    args = parser.parse_args()

    model, results = train(args)

    print(results)
