import os
import random
import numpy as np
import sys

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

import pytorch_lightning as pl

from scipy.stats import norm

CHECKPOINT_PATH = './checkpoints'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)


def set_deterministic():
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def load_latest(trainer, save_name, inference=False, map_location=None, silent = False):
    """Loads the last found model from the checkpoints directory

    Args:
        trainer: PyTorch Lightning module that has the .load_from_checkpoint attribute
        save_name: model name
        inference (bool, optional): whether or not to freeze weights. Defaults to False.
        map_location (device, optional): which device to map the loaded model to. Defaults to None.
        silent (bool, optional): suppresses printing unless no model is found
    """

    def version_to_number(filename):
        return int(filename.rsplit('_', 1)[-1])

    def checkpoint_to_numbers(filename):
        parts = filename.split('=')
        if len(parts) == 3: # epoch=[a]-step=[b].ckpt
            a = int(parts[1][:-5]) # strip '-step'
            b = int(parts[2][:-5]) # strip '.ckpt'
        elif len(parts) == 2: # epoch=[a].ckpt
            a = int(parts[1][:-5]) # strip '.ckpt'
            b = 0
        else:
            return filename
        return (a, b)

    def find_latest_version(save_name):
        save_loc = os.path.join(
            CHECKPOINT_PATH, save_name, 'lightning_logs')
        folders = os.listdir(save_loc)
        if len(folders) == 0: return "None"
        folders.sort(key=version_to_number)
        latest_version = folders[-1]
        checkpoints = os.listdir(os.path.join(save_loc, latest_version, 'checkpoints'))
        if len(checkpoints) == 0: return "None"
        checkpoints.sort(key=checkpoint_to_numbers)
        cpt = checkpoints[-1]
        return os.path.join(save_loc, latest_version, 'checkpoints', cpt)

    pretrained_filename = find_latest_version(save_name)
    if os.path.isfile(pretrained_filename):
        if not silent:
            print("Found pretrained model at %s" % pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = trainer.load_from_checkpoint(
            pretrained_filename, map_location=map_location)
    else:
        sys.exit(f"{save_name} model not found.")

    if inference:
        model.eval()
        model.freeze()

    return model
