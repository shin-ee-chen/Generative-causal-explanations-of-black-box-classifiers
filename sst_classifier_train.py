import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models.sst_bert import SST_Bert_clf
from utils.reproducibility import set_seed, set_deterministic
from datasets.sst import SST

CHECKPOINT_PATH = './checkpoints'

def train(args):
    """
    Inputs:
        args - Namespace object from the argparser
    """

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    if args.add_classes_to_cpt_path == True:
        classes_str = str(5 if args.fine_grained else 3)
        full_log_dir = os.path.join(CHECKPOINT_PATH, args.log_dir + '_' + classes_str)
    else:
        full_log_dir = os.path.join(CHECKPOINT_PATH, args.log_dir)
    os.makedirs(full_log_dir, exist_ok=True)

    device = torch.device('cuda:0' if (torch.cuda.is_available() and args.gpu) else 'cpu')
    train_loader, valid_loader, test_loader = SST.iters(batch_size=args.batch_size,
                                                        fine_grained=True,
                                                        repeat=True,
                                                        device=device)

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=full_log_dir,
                         checkpoint_callback=ModelCheckpoint(
                             save_weights_only=True, mode="max", monitor="Valid acc"),
                         gpus= 1 if (torch.cuda.is_available() and args.gpu) else 0,
                         max_epochs=args.max_epochs,
                         callbacks=[LearningRateMonitor("epoch")],
                         progress_bar_refresh_rate= args.progress_bar_refresh,
                         fast_dev_run=1 if args.debug else False,
                         profiler='simple')

    trainer.logger._default_hp_metric = None

    if args.debug:
        trainer.logger._version = 'debug'

    set_seed(42)
    set_deterministic()

    model = SST_Bert_clf(M=5 if args.fine_grained else 3,
                         dropout=args.dropout,
                         lr=args.lr,
                         layer_decay=args.layer_decay,
                         warm_up=args.warm_up,
                         max_lr_rate=args.max_lr_rate,
                         steps_per_cycle=len(train_loader)
                         )

    # Fit
    trainer.fit(model, train_loader, valid_loader)

    # Eval post training
    model = SST_Bert_clf.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test results
    val_result = trainer.test(model,
                              test_dataloaders=valid_loader,
                              verbose=True)

    test_result = trainer.test(model,
                               test_dataloaders=test_loader,
                               verbose=True)

    result = {"Test": test_result[0]["Test acc"],
              "Valid": val_result[0]["Valid acc"]}

    return model, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--fine_grained', default=True,
                        help='Whether to train on 2 or 5 classes')
    parser.add_argument('--dropout', default=0.1,
                        help='Probability of dropping neuron')

    # Loss and optimizer hyperparameters
    parser.add_argument('--lr', default=2e-5, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Minibatch size')
    parser.add_argument('--max_epochs', default=10, type=int,
                        help='Max number of training epochs')
    parser.add_argument('--layer_decay', default=0.95, type=float,
                        help='Learning rate decay for Bert layers')
    parser.add_argument('--warm_up', default=0.1, type=float,
                        help='Proportion of steps to scale up lr')
    parser.add_argument('--max_lr_rate', default=32, type=float,
                        help='Maximum achieved lr')

    # Other hyperparameters
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders.')
    parser.add_argument('--progress_bar_refresh', default=50,
                        help=('How often to refresh progress bar.'
                              'MUST be 0 for SLURM jobs'))
    parser.add_argument('--log_dir', default='sst_bert', type=str,
                        help='Directory where the PyTorch Lightning logs should be created. Automatically adds \
                            the classes to directory. If not needed, turn off using add_classes_to_cpt_path flag.')
    parser.add_argument('--add_classes_to_cpt_path', default=True,
                        help='Whether to add the classes to cpt directory.')

    # Debug parameters
    parser.add_argument('--debug', default=False,
                        help=('Whether to check debugs, etc.'))
    parser.add_argument('--gpu', default=True, action='store_true',
                        help=('Whether to train on GPU (if available) or CPU'))

    args = parser.parse_args()

    model, results = train(args)
