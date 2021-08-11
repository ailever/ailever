from ailever.investment import fmlops_bs
from .parallelizer import parallelize
from .logger import Logger
from .fmlops_loader_system import Loader

import os
import numpy as np
from numpy import linalg
import pandas as pd


base_dir = dict()
base_dir['root'] = fmlops_bs.local_system.root.name
base_dir['rawdata_repository'] = fmlops_bs.local_system.root.rawdata_repository.name
base_dir['metadata_store'] = fmlops_bs.local_system.root.metadata_store.name
base_dir['feature_store'] = fmlops_bs.local_system.root.feature_store.name
base_dir['model_registry'] = fmlops_bs.local_system.root.model_registry.name
base_dir['source_repotitory'] = fmlops_bs.local_system.root.source_repository.name

logger = Logger()
from_dir = os.path.join(base_dir['root'], base_dir['rawdata_repository'])

def screener(baskets=None, path=None, period=None):

    if not period:
        period = 10
        logger.normal_logger.info(f'PERIOD INPUT REQUIRED - Default Period:{period}')
    
    if not path:
        path = from_dir
        logger.normal_logger.info(f'PATH INPUT REQUIRED - Default Path:{path}')

    if not baskets:            
        serialized_objects = os.listdir(path)
        serialized_object =list(filter(lambda x: x[-3:] == 'csv', serialized_objects))
        baskets_in_dir = list(map(lambda x: x[:-4], serialized_objects))
        baskets = baskets_in_dir 
        logger.normal_logger.info(f'BASKETS INPUT REQUIRED - Default Basket:{baskets} in the directory:{path}.')    

    logger.normal_logger.info(f'UPDATE FOR BASKETS: {baskets}.')
    loader = Loader()
    loader.ohlcv_loader(baskets=baskets, from_dir=path, to_dir=path)
    
    logger.normal_logger.info(f'RECOMMANDATIONS BASED ON LATEST {period} DAYS.')
    
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
    
    prllz.list = list(map(lambda x:mapper[x], np.argsort(recommand)[::-1]))
    logger.normal_logger.info(prllz.list)
    return prllz


