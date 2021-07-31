import os
import numpy as np
from numpy import linalg
import pandas as pd

def reits_screening(period, path):
    print(f'[AILEVER] Recommandations based on latest {period}')
    idx2csv = dict()

    for idx, csv in enumerate(os.listdir(path)):
        idx2csv[idx] = csv
        if idx == 0 :
            base = pd.read_csv(path+csv)['close'][-period:].fillna(method='bfill').fillna(method='ffill').values[:,np.newaxis]
        else:
            appending = pd.read_csv(path+csv)['close'][-period:].fillna(method='bfill').fillna(method='ffill').values[:,np.newaxis]
            base = np.c_[base, appending]

    x, y = np.arange(base.shape[0]), base
    bias = np.ones_like(x)
    X = np.c_[bias, x]

    b = linalg.inv(X.T@X) @ X.T @ y
    yhat = X@b
    recommand = yhat[-1] - yhat[-2]
    return list(map(lambda x: idx2csv[x][:-4], np.argsort(recommand)[::-1]))

