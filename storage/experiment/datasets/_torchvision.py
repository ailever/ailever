# torch
from torch.utils.data import random_split
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# ailever modules
import options


class TorchVisionDataset:
    def __init__(self, options):
        self.options = options
        self.dataset = getattr(datasets, options.dataset_name)(root=options.dataset_savepath, train=True, transform=transforms.ToTensor(), download=True)
        num_dataset = len(self.dataset)
        num_train = int(num_dataset*options.split_rate)
        num_validation = num_dataset - num_train

        self.dataset = random_split(self.dataset, [num_train, num_validation])
        self.train_dataset = self.dataset[0]
        self.validation_dataset = self.dataset[1]
        self.test_dataset = getattr(datasets, options.dataset_name)(root=options.dataset_savepath, train=False, transform=transforms.ToTensor(), download=True)
        
        options.add.x_train_shape = next(iter(self.train_dataset))[0].size()
        options.add.y_train_shape = 1
        options.add.x_validation_shape = next(iter(self.validation_dataset))[0].size()
        options.add.y_validation_shape = 1
        options.add.x_test_shape = next(iter(self.test_dataset))[0].size()
        options.add.y_test_shape = 1

    def type(self, mode='train'):
        x_size = getattr(self.options.add, 'x_'+mode+'_shape')
        y_size = getattr(self.options.add, 'y_'+mode+'_shape')
        print(f'[DATASET][{mode.upper()}] input size : {x_size}')
        print(f'[DATASET][{mode.upper()}] target size : {y_size}')
        if mode == 'train':
            return self.train_dataset
        elif mode == 'validation':
            return self.validation_dataset
        elif mode == 'test':
            return self.test_dataset
