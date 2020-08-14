# built-in / external modules
import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

# torch
import torch
from torch.utils.data import Dataset

# ailever modules
import options


obj = type('obj', (), {})
class SklearnDataset(Dataset):
    def __init__(self, options):
        self.options = options
        self.train_dataset = getattr(datasets, 'load_'+self.options.dataset_name)()

        self.test_dataset = obj()
        self.train_dataset.x, self.test_dataset.x, self.train_dataset.y, self.test_dataset.y = train_test_split(self.train_dataset.data, self.train_dataset.target, test_size=0.3, shuffle=True)
        
        self.validation_dataset = obj()
        self.train_dataset.x, self.validation_dataset.x, self.train_dataset.y, self.validation_dataset.y = train_test_split(self.train_dataset.x, self.train_dataset.y, test_size=0.2, shuffle=True)
        
        self.train_dataset.x = torch.Tensor(self.train_dataset.x)
        self.train_dataset.y = torch.Tensor(self.train_dataset.y)
        self.validation_dataset.x = torch.Tensor(self.validation_dataset.x)
        self.validation_dataset.y = torch.Tensor(self.validation_dataset.y)
        self.test_dataset.x = torch.Tensor(self.test_dataset.x)
        self.test_dataset.y = torch.Tensor(self.test_dataset.y)
        
        self.options.add.x_train_shape = self.train_dataset.x.size()[1:]
        self.options.add.y_train_shape = self.train_dataset.y.size()[1:]
        self.options.add.x_validation_shape = self.validation_dataset.x.size()[1:]
        self.options.add.y_validation_shape = self.validation_dataset.y.size()[1:]
        self.options.add.x_test_shape = self.test_dataset.x.size()[1:]
        self.options.add.y_test_shape = self.test_dataset.y.size()[1:]
        
    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset.y)
        elif self.mode == 'validation':
            return len(self.validation_dataset.y)
        elif self.mode == 'test':
            return len(self.test_dataset.y)
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            x_item = self.train_dataset.x[idx].to(self.options.device)
            y_item = self.train_dataset.y[idx].to(self.options.device)
        elif self.mode == 'validation':
            x_item = self.validation_dataset.x[idx].to(self.options.device)
            y_item = self.validation_dataset.y[idx].to(self.options.device)
        elif self.mode == 'test':
            x_item = self.test_dataset.x[idx].to(self.options.device)
            y_item = self.test_dataset.y[idx].to(self.options.device)
        return x_item, y_item
    
    def type(self, mode='train'):
        self.mode = mode
        x_size = getattr(self.options.add, 'x_'+mode+'_shape')
        y_size = getattr(self.options.add, 'y_'+mode+'_shape')
        print(f'[DATASET][{mode.upper()}] input size : {x_size}')
        print(f'[DATASET][{mode.upper()}] target size : {y_size}')
        return self
