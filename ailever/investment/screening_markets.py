from .parallelizer import parallelize
from . logger_check import ohlcv_update
import os
import numpy as np
from numpy import linalg
import pandas as pd


def screener(baskets=None, path=None, period=None):
    assert period, 'periods input required'
    ohlcv_update(baskets, path)
    print(f'[AILEVER] Recommandations based on latest {period} days.')
    
    prllz = parallelize(baskets=baskets, path=path,
                        object_format='csv',
                        base_column='close',
                        date_column='date',
                        period=period)

    base = prllz.ndarray
    tickers = prllz.pdframe.columns
    mapper = {idx:ticker for idx, ticker in enumerate(tickers)}

    x, y = np.arange(base.shape[0]), base
    bias = np.ones_like(x)
    X = np.c_[bias, x]

    b = linalg.inv(X.T@X) @ X.T @ y
    yhat = X@b
    recommand = yhat[-1] - yhat[-2]
    return list(map(lambda x: mapper[x], np.argsort(recommand)[::-1]))

