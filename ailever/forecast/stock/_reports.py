from ._stattools import regressor, scaler
import numpy as np
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt


class StockReport:
    r"""
    Example:
	>>> from ailever.forecast.stock import krx
	>>> from ailever.forecast.stock import StockReport
	>>> ...
        >>> df = krx.kospi('2020-01-01')
        >>> sr = StockReport(df, filter_period=200, criterion=1.5)
        >>> sr.KRXreport(sr.index[0])
    """
    def __init__(self, df, filter_period=200, criterion=1.5):
        self.df = df

        norm = scaler.standard(self.df[0][-filter_period:])
        yhat = regressor(norm)
        container = yhat[-1,:] - yhat[0,:]
        self.index = np.where(container>=criterion)[0]

        recommended_stock_info = self.df[1].iloc[self.index]
        alert = list(zip(recommended_stock_info.Name.tolist(), recommended_stock_info.Symbol.tolist())); print(alert)


    def KRXreport(self, i=None, long_period=300, short_period=30):
        i_range = list(range(len(self.df[1])))
        assert i in i_range, f'symbol must be in {i_range}'

        if not i : i = self.index[0]
        info = (i, long_period, short_period) # args params
        selected_stock_info = self.df[1].iloc[info[0]]

        plt.figure(figsize=(13,5))
        plt.title(f'{selected_stock_info.Name}({selected_stock_info.Symbol})')
        plt.grid(True)

        X = self.df[0][:, info[0]][-info[1]:]
        norm = scaler.standard(X)
        yhat = regressor(norm)
        Yhat = yhat*X.std(ddof=1) + X.mean(axis=0)
        plt.plot(Yhat[-info[2]:], lw=0.5, label='population')

        X = self.df[0][:, info[0]][-info[2]:]
        _norm = scaler.standard(X)
        _yhat = regressor(_norm)

        x = _norm - _yhat
        x = scaler.minmax(x)

        index = {}
        index['lower'] = np.where((x>=0) & (x<0.2))[0]
        index['upper'] = np.where((x<=1) & (x>0.8))[0]

        plt.plot(X, label='sample')
        plt.plot(index['lower'], X[index['lower']], lw=0, marker='_', label='lower bound')
        plt.plot(index['upper'], X[index['upper']], lw=0, marker='_', label='upper bound')
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
            plt.plot(index['min'], X[index['min']], lw=0, c='red', markersize=10, marker='^', label='Buy')
            plt.plot(index['down'], X[index['down']], lw=0, c='red', alpha=0.3, marker='^', label='Buy')
            plt.plot(index['mid'], X[index['mid']], lw=0, c='green', marker='o', label='base')
            plt.plot(index['up'], X[index['up']], lw=0, c='blue', alpha=0.3, marker='v', label='Sell')
            plt.plot(index['max'], X[index['max']], lw=0, c='blue', markersize=10, marker='v', label='Sell')
        else: # descend field
            plt.plot(index['min'], X[index['min']], lw=0, c='blue', markersize=10, marker='v', label='Sell')
            plt.plot(index['down'], X[index['down']], lw=0, c='blue', alpha=0.3, marker='v', label='Sell')
            plt.plot(index['mid'], X[index['mid']], lw=0, c='green', marker='o', label='base')
            plt.plot(index['up'], X[index['up']], lw=0, c='red', alpha=0.3, marker='^', label='Buy')
            plt.plot(index['max'], X[index['max']], lw=0, c='red', markersize=10, marker='^', label='Buy')

        plt.legend()
        plt.tight_layout()
        plt.show()

        print(selected_stock_info)

