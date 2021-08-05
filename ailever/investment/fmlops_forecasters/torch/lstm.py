import torch
import torch.nn as nn


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

