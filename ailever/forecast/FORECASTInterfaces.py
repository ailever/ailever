from ._dashboard import dashboard

def regressor(norm):
    from ._stattools import regressor
    return regressor(norm=norm)

def TSA(TS, lag=1, select_col=0, visualize=True):
    from ._tsa import TSA
    return TSA(TS=TS, lag=lag, select_col=select_col, visualize=visualize)

def RSDA():
    from .ts_residual import RSDA
    return RSDA()

