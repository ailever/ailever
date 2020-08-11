# built-in / external modules
import json
import pickle
import h5py
import pandas as pd

# torch
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ailever modules
import options


def AileverDataset(options):
    global datasets

    train_dataset = datasets.MNIST(root=options.dataset_path,
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    train_dataset, validation_dataset = random_split(train_dataset, [50000, 10000])
    
    test_dataset = datasets.MNIST(root=options.dataset_path,
                                  train=False,
                                  transform=transforms.ToTensor(),
                                  download=True)

    return train_dataset, validation_dataset, test_dataset


def main(options):
    train_dataset, validation_dataset, test_dataset = AileverDataset(options)
    train_dataloader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=options.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False)

    x_train, y_train = next(iter(train_dataloader))
    x_train, y_train = next(iter(validation_dataloader))
    x_train, y_train = next(iter(test_dataloader))


if __name__ == "__main__":
    options = options.load()
    main(options)
