import torch

import torchvision
from torchvision import transforms
from torchvision.datasets import MNIST

import torch.utils.data as data

def MNIST_limited(root='./Dataset', train=True, labels=[3, 8], train_val_prop=5./6., transform=False):
    """ Takes the regular 10-class MNIST and limits it to a subset, useful for generating the 3 vs. 8 explainer.

    Args:
        root (str, optional): Gives the directory with the MNIST files. Defaults to './Dataset'.
        train (bool, optional): Whether to get train or test set. Defaults to True.
        labels (list, optional): The list with permissible labels. Defaults to [3, 8].
        train_val_prop ([type], optional): The proportion of traindata to all available train data.
            Used for finding the train/val split. Defaults to 5./6. as used in paper.
        transform (boolean, optional): Whether or not to standardize the input data using z-transform.
            Defaults to False, as it is likely not needed for MNIST.

    Returns:
        MNIST dataset: returns the MNIST dataset with only the specified labels.
    """
    
    def find_MNIST_stats():
        """
        Finds the mean and std for the MNIST training set. Useful for whitening the data.
        
        """

        train_set = MNIST('./Dataset', train=True, download=True)

        MNIST_mean = (train_set.data / 255.0).mean(axis=(0, 1, 2))
        MNIST_std = (train_set.data / 255.0).std(axis=(0, 1, 2))

        return MNIST_mean, MNIST_std
    
    MNIST_mean, MNIST_std = find_MNIST_stats()

    if transform == True:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(
                                        MNIST_mean, MNIST_std)
                                        ])
    else:
        transform = transforms.Compose([transforms.ToTensor()])
                                        
    if train:
        dataset = MNIST('./Dataset', train=True, 
                        download=True, transform=transform)
    else:
        dataset = MNIST('./Dataset', train=False,
                        download=True, transform=transform)
        
    idx = [y in labels for y in dataset.targets]
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    
    for new_t, old_t in enumerate(labels):
        dataset.targets[dataset.targets == old_t] = new_t
        
    if train:

        train_range = range(0, int(train_val_prop * len(dataset)))
        valid_range = range(int(train_val_prop * len(dataset)),  len(dataset))

        train_set = torch.utils.data.Subset(dataset, train_range).dataset
        valid_set = torch.utils.data.Subset(dataset, valid_range).dataset
    
        return train_set, valid_set

    return dataset
