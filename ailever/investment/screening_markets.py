from .parallelizer import parallelize
from .logger_check import ohlcv_update
from ._fmlops_policy import fmlops_bs

import os
import numpy as np
from numpy import linalg
import pandas as pd


base_dir = dict()
base_dir['root'] = fmlops_bs.local_system.root
base_dir['rawdata_repository'] = fmlops_bs.local_system.rawdata_repository
base_dir['metadata_store'] = fmlops_bs.local_system.metadata_store
base_dir['feature_store'] = fmlops_bs.local_system.feature_store
base_dir['model_registry'] = fmlops_bs.local_system.model_registry
base_dir['source_repotitory'] = fmlops_bs.local_system.source_repository

dataset_dirname = os.path.join(base_dir['root'], base_dir['rawdata_repository'])

    
def screener(baskets=None, path=dataset_dirname, period=None):
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

