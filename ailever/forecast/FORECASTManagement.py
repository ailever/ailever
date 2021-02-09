def dashboard(name, host='127.0.0.1', port='8050'):
    from ._dashboard import dashboard
    dashboard(name, host='127.0.0.1', port='8050')

def regressor(norm):
    from ._stattools import regressor
    return regressor(norm)

def TSA(TS, lag=1, select_col=0, visualize=True):
    from ._tsa import TSA
    return TSA(TS, lag=1, select_col=0, visualize=True)

def RSDA():
    from .ts_residual import RSDA
    return RSDA()

