from ._stattools import regressor, scaler
from ._deepNN import StockReader, Model, Criterion
import os
import json
from urllib.request import urlretrieve
import numpy as np
from numpy import linalg
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader


dummies = type('dummies', (dict,), {})
class AILF:
    r"""
    Examples:
	>>> from ailever.forecast.stock import krx, AILF
	>>> ...
        >>> Df = krx.kospi('2018-01-01')
        >>> ailf = AILF(Df, filter_period=300, criterion=1.5)
        >>> ailf.KRXreport(ailf.index[0], long_period=200, short_period=30, back_shifting=0, return_Xy=False)

    Examples:
	>>> from ailever.forecast.stock import krx, AILF
	>>> ...
        >>> Df = krx.kospi('2018-01-01')
        >>> ailf = AILF(Df, filter_period=300, criterion=1.5)
        >>> ailf.train(ailf.index[0], epochs=5000, breaking=0.0001, details=False, onlyload=False)
        >>> ailf.KRXreport(ailf.index[0], long_period=200, short_period=30, back_shifting=0, return_Xy=False)

    Examples:
	>>> from ailever.forecast.stock import krx, AILF
	>>> ...
        >>> Df = krx.kospi('2018-01-01')
        >>> ailf = AILF(Df, filter_period=300, criterion=1.5)
        >>> ailf.train(ailf.index[0], onlyload=True)
        >>> ailf.KRXreport(ailf.index[0], long_period=200, short_period=30, back_shifting=0, return_Xy=False)
    """

    def __init__(self, Df, filter_period=300, criterion=1.5):
        if not os.path.isdir('.Log') : os.mkdir('.Log')

        self.deepNN = None
        self.Df = Df

        norm = scaler.standard(self.Df[0][-filter_period:])
        yhat = regressor(norm)
        container = yhat[-1,:] - yhat[0,:]
        self.index = np.where(container>=criterion)[0]

        recommended_stock_info = self.Df[1].iloc[self.index]
        alert = list(zip(recommended_stock_info.Name.tolist(), recommended_stock_info.Symbol.tolist())); print(alert)
        
        self.dummies = dummies()

    def _train_init(self, stock_num=None):
        StockDataset = StockReader(self.Df, stock_num)
        train_dataset = StockDataset.type('train')
        validation_dataset = StockDataset.type('validation')

        self.train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=100, shuffle=False, drop_last=True)

    def train(self, stock_num=None, epochs=2000, breaking=0.0001, details=False, onlyload=False):
        if not stock_num : stock_num = self.index[0]
        self._train_init(stock_num)
        selected_stock_info = self.Df[1].iloc[stock_num]
        symbol = selected_stock_info.Symbol

        if torch.cuda.is_available() : device = torch.device('cuda')
        else : device = torch.device('cpu')
        
        if details:
            h_dim = int(input('* hidden dimension (default 32, int) : '))
            attn_drop = float(input('* drop rate of each attention (default 0.1, float) : '))
            attn_head = int(input('* number of head on each attention : (default 2, int) : '))
            n_layers = int(input('* number of the attention layer (default 1, int) : '))
            model = Model(h_dim, attn_drop, attn_head, n_layers)
        else:
            model = Model()

        if onlyload:
            urlretrieve(f'https://github.com/ailever/openapi/raw/master/forecast/stock/model{symbol}.pth', f'./.Log/model{symbol}.pth')
            model.load_state_dict(torch.load(f'.Log/model{symbol}.pth'))
            print(f'[AILF] The file ".Log/model{symbol}.pth" is successfully loaded!')
            self.deepNN = model
            self.deepNN.stock_info = selected_stock_info
            return None
        else:
            if os.path.isfile(f'.Log/model{symbol}.pth'):
                model.load_state_dict(torch.load(f'.Log/model{symbol}.pth'))
                print(f'[AILF] The file ".Log/model{symbol}.pth" is successfully loaded!')

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
        
        # model save
        self.deepNN = model
        self.deepNN.stock_info = selected_stock_info
        torch.save(model.state_dict(), f'.Log/model{symbol}.pth')
        print(f'[AILF] The file ".Log/model{symbol}.pth" is successfully saved!')
        
        if not os.path.isfile('.Log/model_spec.json'):
            with open('.Log/model_spec.json', 'w') as f:
                json.dump(dict(), f, indent=4)

        with open('.Log/model_spec.json', 'r') as f:
            self.model_spec = json.load(f)

        self.model_spec[f'{symbol}'] = cost.tolist()

        with open('.Log/model_spec.json', 'w') as f:
            json.dump(self.model_spec, f, indent=4)


    def KRXreport(self, i=None, long_period=200, short_period=30, back_shifting=0, return_Xy=False):
        i_range = list(range(len(self.Df[1])))
        assert i in i_range, f'symbol must be in {i_range}'

        if not i : i = self.index[0]
        info = (i, long_period, short_period, back_shifting) # args params
        selected_stock_info = self.Df[1].iloc[info[0]]
        symbol = selected_stock_info.Symbol

        urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/forecast/stock/model_spec.json', './.Log/model_spec.json')
        with open('.Log/model_spec.json', 'r') as f:
            self.model_spec = json.load(f)

        ##########################################################################

        _, axes = plt.subplots(4,1,figsize=(13,15))
        axes[0].grid(True)
        axes[1].grid(True)
        axes[2].grid(True)
        axes[3].grid(True)

        axes[0].set_title(f'{selected_stock_info.Name}({selected_stock_info.Symbol})')

        try:
            df = fdr.DataReader(symbol)
            X = df.Close.values[-info[1]:]
        except:
            X = self.Df[0][:, info[0]][-info[1]:]
        norm = scaler.standard(X)
        yhat = regressor(norm)
        Yhat = yhat*X.std(ddof=1) + X.mean(axis=0)
        axes[0].plot(Yhat[-info[2]:], lw=0.5, label='longterm-trend')

        try:
            df = fdr.DataReader(symbol)
            if info[3] == 0:
                X = df.Close.values[-info[2]:]
                X_High = df.High.values[-info[2]:]
                X_Low = df.Low.values[-info[2]:]
                X_Open = df.Open.values[-info[2]:]
            elif info[3] > 0:
                X = df.Close.values[-info[3]-info[2]:-info[3]]
                X_High = df.High.values[-info[3]-info[2]:-info[3]]
                X_Low = df.Low.values[-info[3]-info[2]:-info[3]]
                X_Open = df.Open.values[-info[3]-info[2]:-info[3]]
        except:
            if info[3] == 0:
                X = self.Df[0][:, info[0]][-info[2]:]
            elif info[3] > 0:
                X = self.Df[0][:, info[0]][-info[3]-info[2]:-info[3]]

        _norm = scaler.standard(X)
        _yhat = regressor(_norm)

        x = _norm - _yhat
        x = scaler.minmax(x)

        index = {}
        index['lower'] = np.where((x>=0) & (x<0.2))[0]
        index['upper'] = np.where((x<=1) & (x>0.8))[0]

        try:
            from scipy import stats
            data = X_High - X_Low
            interval = stats.t.interval(alpha=0.99, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data)) 
            itv = (interval[1] - interval[0])/2

            axes[0].fill_between(x=range(len(X_Open)), y1=X_High, y2=X_Low, alpha=0.7, color='lightgray')
            axes[0].fill_between(x=range(len(X)), y1=X+itv, y2=X-itv, alpha=0.3, color='green')
            axes[0].text(len(X_Open)*0.8, X_Open[-1]+itv, f'Upper-Bound:{int(X_Open[-1]+itv)}')
            axes[0].text(len(X_Open)*0.8, X_Open[-1]-itv, f'Lower-Bound:{int(X_Open[-1]-itv)}')
            axes[0].plot(X_Open, color='green', label='TS(Open)')
            axes[0].plot(X, color='green', lw=3, label='TS(Close)')
            axes[0].plot(index['lower'], X[index['lower']], lw=0, marker='_', label='Lower Bound')
            axes[0].plot(index['upper'], X[index['upper']], lw=0, marker='_', label='Upper Bound')
            axes[0].plot(_yhat*X.std(ddof=1) + X.mean(axis=0), c='orange', label='shortterm-trend')
        except:
            axes[0].plot(X, label='time-series')
            axes[0].plot(index['lower'], X[index['lower']], lw=0, marker='_', label='Lower Bound')
            axes[0].plot(index['upper'], X[index['upper']], lw=0, marker='_', label='Upper Bound')
            axes[0].plot(_yhat*X.std(ddof=1) + X.mean(axis=0), c='orange', label='shortterm-trend')

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
            axes[0].plot(index['min'], X[index['min']], lw=0, c='red', markersize=10, marker='^', label='Strong Buy')
            axes[0].plot(index['down'], X[index['down']], lw=0, c='red', alpha=0.3, marker='^', label='Weak Buy')
            axes[0].plot(index['mid'], X[index['mid']], lw=0, c='green', marker='o', label='base')
            axes[0].plot(index['up'], X[index['up']], lw=0, c='blue', alpha=0.3, marker='v', label='Weak Sell')
            axes[0].plot(index['max'], X[index['max']], lw=0, c='blue', markersize=10, marker='v', label='Strong Sell')
        else: # descend field
            axes[0].plot(index['min'], X[index['min']], lw=0, c='blue', markersize=10, marker='v', label='Strong Sell')
            axes[0].plot(index['down'], X[index['down']], lw=0, c='blue', alpha=0.3, marker='v', label='Weak Sell')
            axes[0].plot(index['mid'], X[index['mid']], lw=0, c='green', marker='o', label='base')
            axes[0].plot(index['up'], X[index['up']], lw=0, c='red', alpha=0.3, marker='^', label='Weak Buy')
            axes[0].plot(index['max'], X[index['max']], lw=0, c='red', markersize=10, marker='^', label='Strong Buy')

        axes[0].legend()

        
        slopes1 = []
        slopes2 = []
        for shifting in range(0,len(self.Df[0])):
            if shifting+info[2] > len(self.Df[0])-1: break

            if shifting == 0 :
                x = np.arange(len(self.Df[0][:,info[0]][-info[2]:]))
                y = self.Df[0][:,info[0]][-info[2]:]
            else:
                x = np.arange(len(self.Df[0][:,info[0]][-shifting-info[2]:-shifting]))
                y = self.Df[0][:,info[0]][-shifting-info[2]:-shifting]
            bias = np.ones_like(x)
            X = np.c_[bias, x]

            b = linalg.inv(X.T@X) @ X.T @ y
            yhat = X@b
            slopes1.append((yhat[-1] - yhat[0])/(info[2]-1))
            slopes2.append((y[-1] - y[0])/(info[2]-1))
        
        self.dummies['KRXreport'] = dict()
        self.dummies['KRXreport']['slopes1'] = slopes1
        self.dummies['KRXreport']['slopes2'] = slopes2

        axes[1].plot(self.Df[0][:,info[0]])
        axes[1].axvline(len(self.Df[0])-info[3]-info[2]*(-1), c='red')
        axes[1].axvline(len(self.Df[0])-info[3]-info[2]*0, ls=':', c='red')
        axes[1].axvline(len(self.Df[0])-info[3]-info[2]*2, ls=':', c='red')
        if back_shifting == 0 : sp = self.Df[0][:,info[0]][-info[2]:].mean()
        else : sp = self.Df[0][:,info[0]][-info[3]-info[2]:-info[3]].mean()
        axes[1].plot([len(self.Df[0])-info[3]-info[2]*1, len(self.Df[0])-info[3]-info[2]*0], [sp,sp], c='black')
        axes[1].text(len(self.Df[0])-info[3]-info[2]*1, sp, f'S.P.:{info[2]}')

        axes[2].plot(slopes1[::-1])
        axes[2].axhline(0, ls=':', c='black')
        axes[2].axvline(len(self.Df[0])-info[3]-info[2]*0, c='red')
        axes[2].axvline(len(self.Df[0])-info[3]-info[2]*1, ls=':', c='red')
        axes[2].axvline(len(self.Df[0])-info[3]-info[2]*2, ls=':', c='red')
        axes[2].plot([len(self.Df[0])-info[3]-info[2]*2, len(self.Df[0])-info[3]-info[2]*1], [0,0], c='black')
        axes[2].text(len(self.Df[0])-info[3]-info[2]*2, 0, f'S.P.:{info[2]}')

        axes[3].plot(slopes2[::-1])
        axes[3].axhline(0, ls=':', c='black')
        axes[3].axvline(len(self.Df[0])-info[3]-info[2]*0, c='red')
        axes[3].axvline(len(self.Df[0])-info[3]-info[2]*1, ls=':', c='red')
        axes[3].axvline(len(self.Df[0])-info[3]-info[2]*2, ls=':', c='red')
        axes[3].plot([len(self.Df[0])-info[3]-info[2]*2, len(self.Df[0])-info[3]-info[2]*1], [0,0], c='black')
        axes[3].text(len(self.Df[0])-info[3]-info[2]*2, 0, f'S.P.:{info[2]}')


        plt.tight_layout()
        plt.show()
        print(selected_stock_info)
        
        if torch.cuda.is_available() : device = torch.device('cuda')
        else : device = torch.device('cpu')
        _ont = 2*(x - 0.5)
        xset = np.c_[_norm, _ont]
        xset = torch.from_numpy(xset).type(torch.FloatTensor).unsqueeze(0).to(device)


        # when there exist the file '.Log/model{~}.pth',
        try: 
            # when self.KRXreport() is called after self.train()
            if self.deepNN: 
                # when first argument(self.train(-, ~) = self.KRXreport(-, ~)) is same,
                if self.deepNN.stock_info.Symbol == symbol: 
                    prob = self.deepNN(xset).squeeze()
                    print('Probability :', prob)
                    print('Cost :', self.model_spec[f'{symbol}'])
                # when first argument(self.train(-, ~) = self.KRXreport(-, ~)) is different,
                else: 
                    urlretrieve(f'https://github.com/ailever/openapi/raw/master/forecast/stock/model{symbol}.pth', f'./.Log/model{symbol}.pth')
                    self.deepNN = Model()
                    self.deepNN.load_state_dict(torch.load(f'.Log/model{symbol}.pth', map_location=torch.device(device)))
                    self.deepNN.stock_info = selected_stock_info
                    prob = self.deepNN(xset).squeeze()
                    print('Probability :', prob)
                    print('Cost :', self.model_spec[f'{symbol}'])

            # when self.KRXreport() is called before self.train()
            elif os.path.isfile(f'.Log/model{symbol}.pth'):
                self.deepNN = Model()
                self.deepNN.load_state_dict(torch.load(f'.Log/model{symbol}.pth', map_location=torch.device(device)))
                self.deepNN.stock_info = selected_stock_info
                prob = self.deepNN(xset).squeeze()
                print('Probability :', prob)
                print('Cost :', self.model_spec[f'{symbol}'])

            # when self.KRXreport() is called before self.train()
            else: 
                urlretrieve(f'https://github.com/ailever/openapi/raw/master/forecast/stock/model{symbol}.pth', f'./.Log/model{symbol}.pth')
                self.deepNN = Model()
                self.deepNN.load_state_dict(torch.load(f'.Log/model{symbol}.pth', map_location=torch.device(device)))
                self.deepNN.stock_info = selected_stock_info
                prob = self.deepNN(xset).squeeze()
                print('Probability :', prob)
                print('Cost :', self.model_spec[f'{symbol}'])

        # when there not exist the file 'model{~}.pth',
        except: 
            prob = None
            print('Probability :', prob)

        if return_Xy:
            return xset, prob



    def TSA(self, i=None, long_period=200, short_period=5, back_shifting=3):
        if not i:
            i = self.index[0]
	info = (i, long_period, short_period, back_shifting)
	selected_stock_info = self.Df[1].iloc[info[0]]

	_, axes = plt.subplots(4,1, figsize=(13,10))
	if back_shifting == 0 :
	    x = self.Df[0][:, info[0]][-info[2]:]
	else:
	    x = self.Df[0][:, info[0]][-info[3]-info[2]:-info[3]]
	x_ = np.sort(x)
	index1 = np.where(x == x_)[0]
	index2 = np.where(x == x_[::-1])[0]

	axes[0].plot(x_, c='red', marker='o')
	axes[0].plot(x_[::-1], c='blue', marker='o')
	axes[0].plot(x, lw=5, c='black', marker='o')
	axes[0].plot(index1, x[index1], color='red', lw=0, marker='^')
	axes[0].plot(index2, x[index2], color='blue', lw=0, marker='v')
	axes[0].axhline((x_[0]+x_[-1])/2, ls=':')
	axes[0].axvline(info[2]-1, ls=':', c='black')
	axes[0].grid(True)
	axes[0].set_title(f'{selected_stock_info.Name}({selected_stock_info.Symbol})')

	if -info[3]+info[2] >= 0:
	    y = self.Df[0][:, info[0]][-info[3]-info[2]:]
	else:
	    y = self.Df[0][:, info[0]][-info[3]-info[2]:-info[3]+info[2]]
	axes[1].plot(x, lw=5, c='black', marker='o')
	axes[1].plot(y, marker='o')
	axes[1].axvline(info[2]-1, ls=':', c='black')
	axes[1].grid(True)

	x = np.diff(x)
	x_ = np.sort(x)
	index1 = np.where(x == x_)[0]
	index2 = np.where(x == x_[::-1])[0]
	axes[2].plot(x_, c='red', marker='o')
	axes[2].plot(x_[::-1], c='blue', marker='o')
	axes[2].plot(x, c='black', lw=5, marker='o')
	axes[2].plot(index1, x[index1], color='red', lw=0, marker='^')
	axes[2].plot(index2, x[index2], color='blue', lw=0, marker='v')
	axes[2].axhline(0, ls=':', c='black')
	axes[2].axvline(info[2]-2, ls=':', c='black')
	axes[2].grid(True)

	y = np.diff(y)
	axes[3].plot(x, lw=5, c='black', marker='o')
	axes[3].plot(y, marker='o')
	axes[3].axvline(info[2]-2, ls=':', c='black')
	axes[3].axhline(0, ls=':', c='black')
	axes[3].grid(True)

	plt.tight_layout()

