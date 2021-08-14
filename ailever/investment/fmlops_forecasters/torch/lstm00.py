import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from ailever.investment import __fmlops_bs__ as fmlops_bs
from ailever.investment import Loader

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# train_specification
adjustable_conditions = ['device', 'batch_size', 'shuffle', 'drop_last', 'epochs']
retrainable_conditions = ['architecture', 'ticker', 'base_columns', 'packet_size', 'prediction_interval', 'start', 'end']

def UI_Transformation(specification):
    # initializing frmaework for transfer to fmlops_managers
    specification['framework'] = 'torch'
    return specification


class Scaler:
    def standard(self, X, inverse=False, return_statistics=False):
        if not inverse:
            self.mean = X.mean(dim=0, keepdim=True)
            self.std = X.std(dim=0, keepdim=True)
            X = (X - self.mean)/ self.std
            if return_statistics:
                return X, (self.mean, self.std)
            else:
                return X
        else:
            X = X * self.std + self.mean
            if return_statistics:
                return X, (self.mean, self.std)
            else:
                return X

    def minmax(self, X, inverse=False, return_statistics=False):
        if not inverse:
            self.min = X.min(dim=0, keepdim=True).values
            self.max = X.max(dim=0, keepdim=True).values
            X = (X - self.min)/ (self.max - self.min)
            if return_statistics:
                return X, (self.max, self.min)
            else:
                return X
        else:
            X = X*(self.max - self.min) + self.min
            if return_statistics:
                return X, (self.min, self.max)
            else:
                return X

    def fft(self, X, inverse=False):
        if not inverse:
            X = torch.fft.fft(X)
            return X
        else:
            X = torch.fft.ifft(X)
            return X



class InvestmentDataset(Dataset):
    def __init__(self, train_specification):
        self.loader = Loader()

        ticker = train_specification['ticker']
        self.device = train_specification['device']
        self.packet_size = train_specification['packet_size']
        self.prediction_interval = train_specification['prediction_interval']
        self.base_columns = train_specification['base_columns']
        self.train_range = self.packet_size - self.prediction_interval
        
        self.frame = self.loader.ohlcv_loader(baskets=[ticker]).dict[ticker].reset_index()[self.base_columns]
        self.frame.date = pd.to_datetime(self.frame.date.astype('str'))
        self.frame = self.frame.set_index('date')

        self.frame_train = self.frame.loc[train_specification['start']:train_specification['end']]
        self.frame_test = self.frame.loc[train_specification['end']:]
        self.frame_last_packet = self.frame.iloc[-self.packet_size:]
        self.tensor_train = torch.from_numpy(self.frame_train.values)
        self.tensor_test = torch.from_numpy(self.frame_test.values)
        self.tensor_last_packet = torch.from_numpy(self.frame_last_packet.values)

    def __len__(self):
        if self.mode == 'train':
            return self.tensor_train.size()[0] - self.packet_size
        elif self.mode == 'test':
            return self.tensor_test.size()[0] - self.packet_size

    def __getitem__(self, idx):
        S = Scaler()
        if self.mode == 'train':
            time_series = S.standard(self.tensor_train[idx:idx+self.packet_size])
        elif self.mode == 'test':
            time_series = S.standard(self.tensor_test[idx:idx+self.packet_size])

        x_item = time_series[:self.train_range].to(self.device)
        y_item = time_series[self.train_range:].to(self.device)
        return x_item, y_item

    def type(self, mode='train'):
        self.mode = mode
        self.frame = getattr(self, f'frame_{mode}')
        return self



def InvestmentDataLoader(train_specification):
    dataset = InvestmentDataset(train_specification)
    batch_size = train_specification['batch_size']
    shuffle = train_specification['shuffle']
    drop_last = train_specification['drop_last']
    train_dataloader = DataLoader(dataset.type('train'), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    test_dataloader = DataLoader(dataset.type('test'), batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return train_dataloader, test_dataloader



class Model(nn.Module):
    def __init__(self, train_specification):
        super(Model, self).__init__()
        num_features = len(train_specification['base_columns']) - 1
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=1024, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(1024, 1024)
        self.linear1_1 = nn.Linear(1024, 1024)
        self.linear1_2 = nn.Linear(1024, 1024)
        self.linear1_3 = nn.Linear(1024, 1024)
        self.linear1_4 = nn.Linear(1024, 1024)
        self.linear1_5 = nn.Linear(1024, 1024)
        self.linear1_6 = nn.Linear(1024, num_features)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)

        self.linear2 = nn.Linear(265, 100)
        self.batch_norm = nn.BatchNorm1d(100)

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = self.linear1(x)
        x = self.linear2(x.transpose(1,2)).transpose(1,2)
        
        x = self.linear1_1(self.relu(x))
        x = self.linear1_2(self.relu(x))
        x = self.linear1_3(self.relu(x))
        x = self.linear1_4(self.relu(x))
        x = self.linear1_5(self.relu(x))
        x = self.linear1_6(self.batch_norm(x))
        return x



class Criterion(nn.Module):
    def __init__(self, train_specification):
        super(Criterion, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, hypothesis, target):
        return self.mse(hypothesis, target)



def Optimizer(model, train_specification):
    optimizer = optim.Adamax(model.parameters())
    return optimizer


