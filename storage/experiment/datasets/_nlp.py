# built-in / external modules
from nlp import load_dataset

# torch
from torch.utils.data import Dataset

# ailever modules
import options


obj = type('obj', (), {})
class NLPDataset(Dataset):
    def __init__(self, options):
        self.options = options
        self.dataset = load_dataset(self.options.dataset_name, cache_dir=self.options.dataset_savepath)
        self.train_dataset = self.dataset['train'];     print(self.train_dataset._info.features)
        self.test_dataset = self.dataset['validation']; print(self.test_dataset._info.features)
        self.validation_dataset = obj()

        self.train_dataset.x = None
        self.train_dataset.y = None
        self.validation_dataset.x = None
        self.validation_dataset.y = None
        self.test_dataset.x = None
        self.test_dataset.y = None

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

