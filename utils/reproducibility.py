import os
import random
import numpy as np 

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
    

def set_deteministic():
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
    
    
def load_latest(trainer, save_name, inference=False, map_location=None):
    """Loads the last found model from the checkpoints directory

    Args:
        trainer: PyTorch Lightning module that has the .load_from_checkpoint attribute
        save_name: model name
        inference (bool, optional): whether or not to freeze weights. Defaults to False.
        map_location (device, optional): which device to map the loaded model to. Defaults to None.
    """

    def find_latest_version(save_name):
        save_loc = os.path.join(
            CHECKPOINT_PATH, save_name, 'lightning_logs')
        latest_version = os.listdir(save_loc)[-1]
        print(os.path.join(save_loc, latest_version, 'checkpoints'), 
              os.listdir(os.path.join(save_loc, latest_version, 'checkpoints')))
        cpt = os.listdir(os.path.join(
            save_loc, latest_version, 'checkpoints'))[-1]

        return os.path.join(save_loc, latest_version, 'checkpoints', cpt)

    pretrained_filename = find_latest_version(save_name)
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s" %
                pretrained_filename)
        # Automatically loads the model with the saved hyperparameters
        model = trainer.load_from_checkpoint(
            pretrained_filename, map_location=map_location)
    else:
        print("No model found")
        return None

    if inference:
        model.eval()
        model.freeze()

    return model
