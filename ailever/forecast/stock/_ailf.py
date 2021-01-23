from ._stattools import regressor, scaler
from ._deepNN import StockReader, Model, Criterion
import os
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


class AILF:
    r"""
    Example:
	>>> from ailever.forecast.stock import krx, AILF
	>>> ...
        >>> Df = krx.kospi('2020-01-01')
        >>> ailf = AILF(Df, filter_period=300, criterion=1.5)
        >>> ailf.train(ailf.index[0], epochs=5000, breaking=0.0001, onlyload=False)
        >>> ailf.KRXreport(ailf.index[0], long_period=200, short_period=30, return_Xy=False)
    """
    def __init__(self, Df, filter_period=300, criterion=1.5):
        self.deepNN = None
        self.Df = Df

        norm = scaler.standard(self.Df[0][-filter_period:])
        yhat = regressor(norm)
        container = yhat[-1,:] - yhat[0,:]
        self.index = np.where(container>=criterion)[0]

        recommended_stock_info = self.Df[1].iloc[self.index]
        alert = list(zip(recommended_stock_info.Name.tolist(), recommended_stock_info.Symbol.tolist())); print(alert)
        
    def _train_init(self, stock_num=None):
        if not stock_num : stock_num = self.index[0]
        StockDataset = StockReader(self.Df, stock_num)
        train_dataset = StockDataset.type('train')
        validation_dataset = StockDataset.type('validation')
        self.train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=100, shuffle=False, drop_last=True)

    def train(self, stock_num=None, epochs=5000, breaking=0.0001, onlyload=False, details=False):
        self._train_init(stock_num)
        selected_stock_info = self.Df[1].iloc[stock_num]
        symbol = selected_stock_info.Symbol

        if not os.path.isdir('.Log') : os.mkdir('.Log')
        if torch.cuda.is_available() : device = torch.device('cuda')
        else : device = torch.device('cpu')
        
        if details:
            h_dim = int(input('* hidden dimension : '))
            attn_drop = int(input('* drop rate of each attention : '))
            attn_head = int(input('* number of head on each attention : '))
            nlayers = int(input('* number of the attention layer : '))
            model = Model(h_dim, attn_drop, attn_head, nlayers)
        else:
            model = Model()

        if os.path.isfile(f'.Log/model{symbol}.pth'):
            model.load_state_dict(torch.load(f'.Log/model{symbol}.pth'))
            print(f'[AILF] The file ".Log/model{symbol}.pth" is successfully loaded!')
            if onlyload:
                self.deepNN = model
                return None
            else:
                print(f'[AILF] The options onlyload will not execute, because the file ".Log/model{symbol}.pth" do not exist!')

        model = model.to(device)
        criterion = Criterion().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-6)

        for epoch in range(epochs):
            # Training
            for batch_idx, (x_train, y_train) in enumerate(self.train_dataloader):
                # forward
                hypothesis = model(x_train)
                cost = criterion(hypothesis, y_train)
                
                # backward
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                if epoch%100 == 0:
                    print(f'[TRAIN][{epoch}/{epochs}] :', cost)
            if cost < breaking : break

            # Validation
            with torch.no_grad():
                model.eval()
                for batch_idx, (x_train, y_train) in enumerate(self.validation_dataloader):
                    # forward
                    hypothesis = model(x_train)
                    cost = criterion(hypothesis, y_train)
                    
                    if epoch%100 == 0:
                        print(f'[VAL][{epoch}/{epochs}] :', cost)
            
        self.deepNN = model
        torch.save(model.state_dict(), f'.Log/model{symbol}.pth')
        print(f'[AILF] The file ".Log/model{symbol}.pth" is successfully saved!')


    def KRXreport(self, i=None, long_period=200, short_period=30, return_Xy=False):
        i_range = list(range(len(self.Df[1])))
        assert i in i_range, f'symbol must be in {i_range}'

        if not i : i = self.index[0]
        info = (i, long_period, short_period) # args params
        selected_stock_info = self.Df[1].iloc[info[0]]

        plt.figure(figsize=(13,5))
        plt.title(f'{selected_stock_info.Name}({selected_stock_info.Symbol})')
        plt.grid(True)

        X = self.Df[0][:, info[0]][-info[1]:]
        norm = scaler.standard(X)
        yhat = regressor(norm)
        Yhat = yhat*X.std(ddof=1) + X.mean(axis=0)
        plt.plot(Yhat[-info[2]:], lw=0.5, label='longterm-trend')

        X = self.Df[0][:, info[0]][-info[2]:]
        _norm = scaler.standard(X)
        _yhat = regressor(_norm)

        x = _norm - _yhat
        x = scaler.minmax(x)

        index = {}
        index['lower'] = np.where((x>=0) & (x<0.2))[0]
        index['upper'] = np.where((x<=1) & (x>0.8))[0]

        plt.plot(X, label='shortterm-trend')
        plt.plot(index['lower'], X[index['lower']], lw=0, marker='_', label='Lower Bound')
        plt.plot(index['upper'], X[index['upper']], lw=0, marker='_', label='Upper Bound')
        plt.plot(_yhat*X.std(ddof=1) + X.mean(axis=0))

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

        index = {}
        index['min'] = np.where((x>=0) & (x<0.1))[0]
        index['down'] = np.where((x>=0.1) & (x<0.45))[0]
        index['mid'] = np.where((x>=0.45)&(x<0.55))[0]
        index['up'] = np.where((x<0.9) & (x>=0.55))[0]
        index['max'] = np.where((x<=1) & (x>=0.9))[0]
        if _yhat[-1] - _yhat[0] > 0: # ascend field
            plt.plot(index['min'], X[index['min']], lw=0, c='red', markersize=10, marker='^', label='Strong Buy')
            plt.plot(index['down'], X[index['down']], lw=0, c='red', alpha=0.3, marker='^', label='Weak Buy')
            plt.plot(index['mid'], X[index['mid']], lw=0, c='green', marker='o', label='base')
            plt.plot(index['up'], X[index['up']], lw=0, c='blue', alpha=0.3, marker='v', label='Weak Sell')
            plt.plot(index['max'], X[index['max']], lw=0, c='blue', markersize=10, marker='v', label='Strong Sell')
        else: # descend field
            plt.plot(index['min'], X[index['min']], lw=0, c='blue', markersize=10, marker='v', label='Strong Sell')
            plt.plot(index['down'], X[index['down']], lw=0, c='blue', alpha=0.3, marker='v', label='Weak Sell')
            plt.plot(index['mid'], X[index['mid']], lw=0, c='green', marker='o', label='base')
            plt.plot(index['up'], X[index['up']], lw=0, c='red', alpha=0.3, marker='^', label='Weak Buy')
            plt.plot(index['max'], X[index['max']], lw=0, c='red', markersize=10, marker='^', label='Strong Buy')

        plt.legend()
        plt.tight_layout()
        plt.show()
        print(selected_stock_info)
        
        if torch.cuda.is_available() : device = torch.device('cuda')
        else : device = torch.device('cpu')
        _ont = 2*(x - 0.5)
        xset = np.c_[_norm, _ont]
        xset = torch.from_numpy(xset).type(torch.FloatTensor).unsqueeze(0).to(device)
        
        if self.deepNN:
            prob = self.deepNN(xset).squeeze()
            print('Probability :', prob)

        else:
            prob = None
            print('Probability :', prob)

        if return_Xy:
            return xset, prob

