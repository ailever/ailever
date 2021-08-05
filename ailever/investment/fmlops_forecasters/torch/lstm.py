import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class InvestmentDataset(Dataset):
    def __init__(self, training_info):
        ticker = training_info['ticker']
        df = pd.read_csv(f'temp/{ticker}.csv')[['Date', 'Close',  'Volume_PCT', 'VIX_Rate', 'VIX_Close_PCT', 'US10YT_Rate', 'Price_Rate']]
        df.Date = pd.to_datetime(df.Date.astype('str'))
        df = df.set_index('Date')

        self.device = training_info['device']
        self.packet_size = 365
        self.predict_range = 100
        self.train_range = self.packet_size - self.predict_range
        self.feature_weights = dict()
        self.feature_weights['Volume'] = 0.01

        self.frame = df
        self.frame_train = df.iloc[:1500]
        self.frame_test = df.iloc[1500:]
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
        S = _Scaler()
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

def InvestmentDataLoader(dataset):
    train_dataloader = DataLoader(dataset.type('train'), , batch_size=batch_size, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(dataset.type('test'), , batch_size=batch_size, shuffle=True, drop_last=False)
    return train_dataloader, test_dataloader

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(input_size=6, hidden_size=1024, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(1024, 1024)
        self.linear1_1 = nn.Linear(1024, 1024)
        self.linear1_2 = nn.Linear(1024, 1024)
        self.linear1_3 = nn.Linear(1024, 1024)
        self.linear1_4 = nn.Linear(1024, 1024)
        self.linear1_5 = nn.Linear(1024, 1024)
        self.linear1_6 = nn.Linear(1024, 6)
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
    def __init__(self):
        super(Criterion, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, hypothesis, target):
        return self.mse(hypothesis, target)


def Optimizer(model):
    optimizer = optim.Adamax(model.parameters())
    return optimizer


