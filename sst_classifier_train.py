import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torchtext

from models.sst_bilstm_cnn import sst_bilstm_cnn
from utils.reproducibility import set_seed, set_deterministic
from datasets.sst import SST, get_glove_url

CHECKPOINT_PATH = './checkpoints'

def train(args):
    """
    Inputs:
        args - Namespace object from the argparser
    """

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    if args.add_classes_to_cpt_path == True:
        classes_str = str(5 if args.fine_grained else 2)
        full_log_dir = os.path.join(CHECKPOINT_PATH, args.log_dir + '_' + classes_str)
    else:
        full_log_dir = os.path.join(CHECKPOINT_PATH, args.log_dir)
    os.makedirs(full_log_dir, exist_ok=True)

    device = torch.device('cuda:0' if (torch.cuda.is_available() and args.gpu) else 'cpu')

    vectors = torchtext.vocab.Vectors(name='glove.840B.300d.sst.txt',
                                      cache='./datasets/SST',
                                      url=get_glove_url()
                                      )

    (train_loader, valid_loader, test_loader), (vocab, train_data) = SST.iters(batch_size=args.batch_size, repeat=True,
                                                                               fine_grained=args.fine_grained, vectors=vectors,
                                                                               device=device, pad_to_max=True)

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=full_log_dir,
                         checkpoint_callback=ModelCheckpoint(
                             save_weights_only=True, mode="max", monitor="Valid acc"),
                         gpus= 1 if (torch.cuda.is_available() and args.gpu) else 0,
                         max_epochs=args.max_epochs,
                         callbacks=[LearningRateMonitor("epoch")],
                         progress_bar_refresh_rate= args.progress_bar_refresh,
                         fast_dev_run=args.debug
                         )

    trainer.logger._default_hp_metric = None

    if args.debug:
        trainer.logger._version = 'debug'

    set_seed(42)
    set_deterministic()

    model = sst_bilstm_cnn(vocab=vocab,
                           dropout=args.dropout,
                           lstm_hidden=args.lstm_hidden,
                           filters=args.cnn_filters,
                           cnn_ksize=args.cnn_ksize,
                           max_ksize=args.max_ksize,
                           M=5 if args.fine_grained else 2,
                           lr=args.lr,
                           padded_length = 82,
                           lr_decay=args.lr_decay)

    # Fit
    trainer.fit(model, train_loader, valid_loader)

    # Eval post training
    model = sst_bilstm_cnn.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

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

    parser.add_argument('--fine_grained', default=False,
                        help='Whether to train on 2 or 5 classes')
    parser.add_argument('--dropout', default=[0.5, 0.1, 0.5],
                        help='Probability of dropping neurons in the different dropout layers')
    parser.add_argument('--lstm_hidden', default=256, type=int,
                        help='Number of hidden nodes in the BiLSTM')
    parser.add_argument('--cnn_filters', default=32, type=int,
                        help='Number of filters in the CNN layer')
    parser.add_argument('--cnn_ksize', default=3, type=int,
                        help='Kernel size of the convolution layer')
    parser.add_argument('--max_ksize', default=2, type=int,
                        help='Kernel size of the max pool layer')

    # Loss and optimizer hyperparameters
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--lr_decay', default=0.85, type=float,
                        help='Epoch based learning rate decay')
    parser.add_argument('--max_epochs', default=25, type=int,
                        help='Max number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--seed', default=42, type= int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders.')
    parser.add_argument('--progress_bar_refresh', default=1,
                        help=('How often to refresh progress bar. MUST be 0 for SLURM jobs'))
    parser.add_argument('--log_dir', default='sst_lstm_cnn', type=str,
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
