import numpy as np
from numpy import linalg

def regressor(norm):
    x, y = list(range(len(norm))), norm
    bias = np.ones_like(x)
    X = np.c_[bias, x]

    b = linalg.inv(X.T@X) @ X.T @ y
    yhat = X@b
    return yhat

