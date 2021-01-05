"""
This file contains some utility functions that can be used throughout training.

"""

import random
import numpy as np 
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deteministic():
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False
