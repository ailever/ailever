from ._stattools import scaler, regressor
import numpy as np
import statsmodels.tsa.api as smt
import torch
import torch.nn as nn
from torch.utils.data import Dataset

StockData = type('StockData', (dict,), {})
class StockReader(Dataset):
    def __init__(self, Df, specific_stock_num, long_period=200, short_period=30, forecast_period=3):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.Df = Df
        self.dataset = StockData()
        self.dataset['x'] = list()
        self.dataset['y'] = list()
        # ex> specific_stock_num : ailf.index[0]
        info = (specific_stock_num, long_period, short_period, forecast_period)
        self.preprocess(info)

        self.train_dataset = StockData()
        self.validation_dataset = StockData()
        self.test_dataset = StockData()

        setsize = len(self.dataset['y'])
        spliter = int(setsize*0.7)
        self.train_dataset['x'] = self.dataset['x'][:spliter]
        self.train_dataset['y'] = self.dataset['y'][:spliter]
        self.validation_dataset['x'] = self.dataset['x'][spliter:]
        self.validation_dataset['y'] = self.dataset['y'][spliter:]
        self.test_dataset['x'] = self.dataset['x'][spliter:]
        self.test_dataset['y'] = self.dataset['y'][spliter:]
            

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset['y'])
        elif self.mode == 'validation':
            return len(self.validation_dataset['y'])
        elif self.mode == 'test':
            return len(self.test_dataset['y'])
    
    def __getitem__(self, idx):
        if self.mode == 'train':
            x_item = self.train_dataset['x'][idx]
            y_item = self.train_dataset['y'][idx]
        elif self.mode == 'validation':
            x_item = self.validation_dataset['x'][idx]
            y_item = self.validation_dataset['y'][idx]
        elif self.mode == 'test':
            x_item = self.test_dataset['x'][idx]
            y_item = self.test_dataset['y'][idx]

        x_item = torch.from_numpy(x_item).type(torch.FloatTensor).to(self.device)
        y_item = torch.from_numpy(y_item).type(torch.FloatTensor).to(self.device)
        return x_item, y_item
    
    def _preprocess(self, i, short_period, forecast_period):
        X = self.specific_stock[i:i+short_period]
        price = self.specific_stock[i+short_period:i+short_period+forecast_period].max()

        _norm = scaler.standard(X)
        _yhat = regressor(_norm)

        # Correlation Analysis
        def taylor_series(x, coef):
            degree = len(coef) - 1
            value = 0
            for i in range(degree+1):
                value += coef[i]*x**(degree-i)
            return value

        xdata = np.linspace(-10,10,len(_yhat))
        ydata = smt.acf(_norm-_yhat, nlags=len(_yhat))
        degree = 2
        coef = np.polyfit(xdata, ydata, degree) #; print(f'Coefficients: {coef}')

        x = ydata - taylor_series(xdata, coef)
        x = scaler.minmax(x)
        _ont = 2*(x - 0.5)

        xset = np.c_[_norm, _ont]
        yset = X[-1] - price
        if yset > 0 :
            yset = np.array([1.])
        else:
            yset = np.array([0.])

        self.dataset['x'].append(xset)
        self.dataset['y'].append(yset)

    def preprocess(self, info):
        self.specific_stock = self.Df[0][:, info[0]]
        for i in range(len(self.specific_stock)):
            if i+info[2]+3 > len(self.specific_stock)-1 : break
            self._preprocess(i, short_period=info[2], forecast_period=info[3])

    def type(self, mode='train'):
        self.mode = mode
        return self


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.embedding = nn.Linear(2,128)

        encoder_layer = nn.TransformerEncoderLayer(d_model=128, dropout=0.01, dim_feedforward=512, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.linear = nn.Linear(128,1)
        self.sigmoid = nn.Sigmoid()

        self.drop = nn.Dropout(p=0.1)
        self.batch_norm128 = torch.nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.embedding(x)
        x = self.drop(self.batch_norm128(x))
        x = self.transformer_encoder(x)
        x = self.linear(self.batch_norm128(x)).squeeze()
        x = self.sigmoid(x.mean(dim=-1, keepdim=True))
        """
        x = (batch, sequence, x_dim)
        q = (sequence, batch, x_dim)
        Q = (sequence, batch, x_dim)
        Q_t = (batch, sequence)

	x :  torch.Size([10, 30, 2])
	x :  torch.Size([10, 30, 30])
	q :  torch.Size([30, 10, 30])
	Q :  torch.Size([30, 10, 30])
	Q_t :  torch.Size([10, 30, 30])
	out :  torch.Size([10, 30])
	out :  torch.Size([10])
        """
        return x

class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.mse = nn.MSELoss(reduction='mean')

    def forward(self, hypothesis, target):
        return self.mse(hypothesis, target)
