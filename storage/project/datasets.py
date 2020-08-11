# built-in / external modules
import json
import pickle
import h5py
import pandas as pd

# torch
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# ailever modules
import options

obj = type('obj', (), {})

class AileverDataset(Dataset):
    def __init__(self, options):
        self.options = options
        self.file_objs = obj()
        self.items = torch.arange(1000*10).view(1000,10).type(torch.FloatTensor)
        
        #xlsx_obj = pd.read_excel(self.options.xlsx_path)
        #json_obj = json.load(open(self.options.json_path))
        #pkl_obj = pickle.load(open(self.options.pkl_path, 'rb'))
        #hdf5_obj = h5py.File(self.options.hdf5_path, 'r')


    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        x_item = torch.Tensor(self.items[idx, 0:9]).to(self.options.device)
        y_item = torch.Tensor(self.items[idx, 9:10]).to(self.options.device)

        return x_item, y_item
    
    def type(self, mode='train'):
        self.loader_type = mode
        return self

def main(options):
    dataset = AileverDataset(options)
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=False)
        
    x_train, y_train = next(iter(dataloader))

if __name__ == "__main__":
    options = options.load()
    main(options)
