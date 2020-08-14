# torch
import torch
from torch.utils.data import Dataset
import options

obj = type('obj', (), {})
class CustomDataset(Dataset):
    def __init__(self, options):
        self.options = options
        self.sizeofset = 10000

        if self.options.dataset_name == 'internal-univariate-linear-scalar':
            self.dataset = obj()
            self.dataset.xy = torch.arange(0*self.sizeofset*2, 3*self.sizeofset*2).view(3*self.sizeofset,2).type(torch.FloatTensor)
            permutation = torch.randperm(3*self.sizeofset)
            spliter = permutation.split([self.sizeofset,self.sizeofset,self.sizeofset])

            self.train_dataset = obj()
            self.train_dataset.x = self.dataset.xy[spliter[0]][:, 0:1]
            self.train_dataset.y = self.dataset.xy[spliter[0]][:, 1:2]
            
            self.validation_dataset = obj()
            self.validation_dataset.x = self.dataset.xy[spliter[1]][:, 0:1]
            self.validation_dataset.y = self.dataset.xy[spliter[1]][:, 1:2]

            self.test_dataset = obj()
            self.test_dataset.x = self.dataset.xy[spliter[2]][:, 0:1]
            self.test_dataset.y = self.dataset.xy[spliter[2]][:, 1:2]

            self.options.add.x_train_shape = self.train_dataset.x.size()[1:]
            self.options.add.y_train_shape = self.train_dataset.y.size()[1:]
            self.options.add.x_validation_shape = self.validation_dataset.x.size()[1:]
            self.options.add.y_validation_shape = self.validation_dataset.y.size()[1:]
            self.options.add.x_test_shape = self.test_dataset.x.size()[1:]
            self.options.add.y_test_shape = self.test_dataset.y.size()[1:]
        
        elif self.options.dataset_name == 'external-univariate-linear-scalar':
            self.train_dataset = obj()
            self.train_dataset.xy = torch.arange(0*self.sizeofset*2, 1*self.sizeofset*2).view(self.sizeofset,2).type(torch.FloatTensor)
            self.train_dataset.x = self.train_dataset.xy[:, 0:1]
            self.train_dataset.y = self.train_dataset.xy[:, 1:2]
            
            self.validation_dataset = obj()
            self.validation_dataset.xy = torch.arange(1*self.sizeofset*2, 2*self.sizeofset*2).view(self.sizeofset,2).type(torch.FloatTensor)
            self.validation_dataset.x = self.validation_dataset.xy[:, 0:1]
            self.validation_dataset.y = self.validation_dataset.xy[:, 1:2]

            self.test_dataset = obj()
            self.test_dataset.xy = torch.arange(2*self.sizeofset*2, 3*self.sizeofset*2).view(self.sizeofset,2).type(torch.FloatTensor)
            self.test_dataset.x = self.test_dataset.xy[:, 0:1]
            self.test_dataset.y = self.test_dataset.xy[:, 1:2]
        
            self.options.add.x_train_shape = self.train_dataset.x.size()[1:]
            self.options.add.y_train_shape = self.train_dataset.y.size()[1:]
            self.options.add.x_validation_shape = self.validation_dataset.x.size()[1:]
            self.options.add.y_validation_shape = self.validation_dataset.y.size()[1:]
            self.options.add.x_test_shape = self.test_dataset.x.size()[1:]
            self.options.add.y_test_shape = self.test_dataset.y.size()[1:]
        
        elif self.options.dataset_name == 'internal-multivariate-linear-scalar':
            self.dataset = obj()
            self.dataset.xy = torch.arange(0*self.sizeofset*10, 3*self.sizeofset*10).view(3*self.sizeofset,10).type(torch.FloatTensor)
            permutation = torch.randperm(3*self.sizeofset)
            spliter = permutation.split([self.sizeofset,self.sizeofset,self.sizeofset])

            self.train_dataset = obj()
            self.train_dataset.x = self.dataset.xy[spliter[0]][:, 0:9]
            self.train_dataset.y = self.dataset.xy[spliter[0]][:, 9:10]
            
            self.validation_dataset = obj()
            self.validation_dataset.x = self.dataset.xy[spliter[1]][:, 0:9]
            self.validation_dataset.y = self.dataset.xy[spliter[1]][:, 9:10]

            self.test_dataset = obj()
            self.test_dataset.x = self.dataset.xy[spliter[2]][:, 0:9]
            self.test_dataset.y = self.dataset.xy[spliter[2]][:, 9:10]
        
            self.options.add.x_train_shape = self.train_dataset.x.size()[1:]
            self.options.add.y_train_shape = self.train_dataset.y.size()[1:]
            self.options.add.x_validation_shape = self.validation_dataset.x.size()[1:]
            self.options.add.y_validation_shape = self.validation_dataset.y.size()[1:]
            self.options.add.x_test_shape = self.test_dataset.x.size()[1:]
            self.options.add.y_test_shape = self.test_dataset.y.size()[1:]
        
        elif self.options.dataset_name == 'external-multivariate-linear-scalar':
            self.train_dataset = obj()
            self.train_dataset.xy = torch.arange(0*self.sizeofset*10, 1*self.sizeofset*10).view(self.sizeofset,10).type(torch.FloatTensor)
            self.train_dataset.x = self.train_dataset.xy[:, 0:9]
            self.train_dataset.y = self.train_dataset.xy[:, 9:10]
            
            self.validation_dataset = obj()
            self.validation_dataset.xy = torch.arange(1*self.sizeofset*10, 2*self.sizeofset*10).view(self.sizeofset,10).type(torch.FloatTensor)
            self.validation_dataset.x = self.validation_dataset.xy[:, 0:9]
            self.validation_dataset.y = self.validation_dataset.xy[:, 9:10]

            self.test_dataset = obj()
            self.test_dataset.xy = torch.arange(2*self.sizeofset*10, 3*self.sizeofset*10).view(self.sizeofset,10).type(torch.FloatTensor)
            self.test_dataset.x = self.test_dataset.xy[:, 0:9]
            self.test_dataset.y = self.test_dataset.xy[:, 9:10]

            self.options.add.x_train_shape = self.train_dataset.x.size()[1:]
            self.options.add.y_train_shape = self.train_dataset.y.size()[1:]
            self.options.add.x_validation_shape = self.validation_dataset.x.size()[1:]
            self.options.add.y_validation_shape = self.validation_dataset.y.size()[1:]
            self.options.add.x_test_shape = self.test_dataset.x.size()[1:]
            self.options.add.y_test_shape = self.test_dataset.y.size()[1:]
        
        elif self.options.dataset_name == 'internal-multivariate-linear-vector':
            self.dataset = obj()
            self.dataset.xy = torch.arange(0*self.sizeofset*10, 3*self.sizeofset*10).view(3*self.sizeofset,10).type(torch.FloatTensor)
            permutation = torch.randperm(3*self.sizeofset)
            spliter = permutation.split([self.sizeofset,self.sizeofset,self.sizeofset])

            self.train_dataset = obj()
            self.train_dataset.x = self.dataset.xy[spliter[0]][:, 0:7]
            self.train_dataset.y = self.dataset.xy[spliter[0]][:, 7:10]
            
            self.validation_dataset = obj()
            self.validation_dataset.x = self.dataset.xy[spliter[1]][:, 0:7]
            self.validation_dataset.y = self.dataset.xy[spliter[1]][:, 7:10]

            self.test_dataset = obj()
            self.test_dataset.x = self.dataset.xy[spliter[2]][:, 0:7]
            self.test_dataset.y = self.dataset.xy[spliter[2]][:, 7:10]

            self.options.add.x_train_shape = self.train_dataset.x.size()[1:]
            self.options.add.y_train_shape = self.train_dataset.y.size()[1:]
            self.options.add.x_validation_shape = self.validation_dataset.x.size()[1:]
            self.options.add.y_validation_shape = self.validation_dataset.y.size()[1:]
            self.options.add.x_test_shape = self.test_dataset.x.size()[1:]
            self.options.add.y_test_shape = self.test_dataset.y.size()[1:]
        
        elif self.options.dataset_name == 'external-multivariate-linear-vector':
            self.train_dataset = obj()
            self.train_dataset.xy = torch.arange(0*self.sizeofset*10, 1*self.sizeofset*10).view(self.sizeofset,10).type(torch.FloatTensor)
            self.train_dataset.x = self.train_dataset.xy[:, 0:7]
            self.train_dataset.y = self.train_dataset.xy[:, 7:10]
            
            self.validation_dataset = obj()
            self.validation_dataset.xy = torch.arange(1*self.sizeofset*10, 2*self.sizeofset*10).view(self.sizeofset,10).type(torch.FloatTensor)
            self.validation_dataset.x = self.validation_dataset.xy[:, 0:7]
            self.validation_dataset.y = self.validation_dataset.xy[:, 7:10]

            self.test_dataset = obj()
            self.test_dataset.xy = torch.arange(2*self.sizeofset*10, 3*self.sizeofset*10).view(self.sizeofset,10).type(torch.FloatTensor)
            self.test_dataset.x = self.test_dataset.xy[:, 0:7]
            self.test_dataset.y = self.test_dataset.xy[:, 7:10]

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
