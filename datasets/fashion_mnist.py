import torch

import torchvision
from torchvision import transforms
from torchvision.datasets import FashionMNIST

import torch.utils.data as data

def Fashion_MNIST_limited(root='./datasets', train=True, classes=[0, 3, 4], test_val_prop=4./10., transform=False):
    """ Takes the regular 10-class MNIST and limits it to a subset, useful for generating the 3 vs. 8 explainer.

    Args:
        root (str, optional): Gives the directory with the MNIST files. Defaults to './Dataset'.
        train (bool, optional): Whether to get train or test set. Defaults to True.
        classes (list, optional): The list with permissible labels. Defaults to [0, 3, 4]/‘t-shirt/top,’ ‘dress,’ and ‘coat’.
        test_val_prop ([type], optional): The proportion of valid data to all available test data.
            Used for finding the val/test split. Defaults to 6./10. as used in paper.
        transform (boolean, optional): Whether or not to standardize the input data using z-transform.
            Defaults to False, as it is likely not needed for MNIST.

    Returns:
        MNIST dataset: returns the MNIST dataset with only the specified classes.
    """
    def find_MNIST_stats():
        """
        Finds the mean and std for the MNIST training set. Useful for whitening the data.
        
        """

        train_set = FashionMNIST(root, train=True, download=True)

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
        train_set = FashionMNIST(root, train=True,
                        download=True, transform=transform)
        test_dataset = FashionMNIST(root, train=False,
                        download=True, transform=transform)

        train_set = keep_selected_classes(classes, train_set)
        test_dataset = keep_selected_classes(classes, test_dataset)

        valid_range = range(int(test_val_prop * len(test_dataset)),  len(test_dataset))
        valid_set = torch.utils.data.Subset(test_dataset, valid_range)

        return train_set, valid_set

    else:
        test_dataset = FashionMNIST(root, train=False,
                        download=True, transform=transform)
        test_dataset = keep_selected_classes(classes, test_dataset)

        test_range = range(0, int(test_val_prop * len(test_dataset)))
        test_set = torch.utils.data.Subset(test_dataset, test_range)
        
        return test_set
    


def keep_selected_classes(classes, dataset):
    """Remove data from classes not used"""
    idx = [y in classes for y in dataset.targets]
    dataset.data = dataset.data[idx]
    dataset.targets = dataset.targets[idx]
    
    for new_t, old_t in enumerate(classes):
        dataset.targets[dataset.targets == old_t] = new_t
    return dataset
    

if __name__ == '__main__':
    train_set, valid_set = Fashion_MNIST_limited()
    test_set = Fashion_MNIST_limited(train=False)
    print(len(valid_set))
    print(len(test_set))