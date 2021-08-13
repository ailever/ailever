from ailever.investment import __fmlops_bs__ as fmlops_bs
from .parallelizer import parallelize
from ._base_transfer import DataTransferCore
from .logger import Logger
from .fmlops_loader_system import Loader
from .fmlops_loader_system.DataVendor import DataVendor

import os
import numpy as np
import pandas as pd
from numpy import linalg
from datetime import datetime


base_dir = dict()
base_dir['root'] = fmlops_bs.local_system.root.name
base_dir['metadata_store'] = fmlops_bs.local_system.root.metadata_store.name
base_dir['feature_store'] = fmlops_bs.local_system.root.feature_store.name
base_dir['model_registry'] = fmlops_bs.local_system.root.model_registry.name
base_dir['source_repotitory'] = fmlops_bs.local_system.root.source_repository.name

logger = Logger()
from_dir = os.path.join(base_dir['root'], base_dir['feature_store'])
to_dir = os.path.join(base_dir['root'], base_dir['feature_store'])

class Screener(DataTransferCore):
    
    fundamentals_moduels_fromyahooquery_dict = DataVendor.fundamentals_modules_fromyahooquery_dict
    fundamentals_modules_fromyahooquery = DataVendor.fundamentals_modules_fromyahooquery
    fmf = DataVendor.fundamentals_modules_fromyahooquery

    @staticmethod
    def fundamentals_screener(baskets=None, from_dir=None, period=None, modules=None, sort_by=None, to_dir=None, output='list'):
        
        if not period:
            period = 10
            logger.normal_logger.info(f'[SCREENER] PERIOD INPUT REQUIRED - Default Period:{period}')       
        if not from_dir:
            path = from_dir   
            logger.normal_logger.info(f'[SCREENER] FROM_DIR INPUT REQUIRED - Default Path:{path}')
        if not to_dir:
            to_dir = to_dir
            logger.normal_logger.info(f'[SCREENER] TO_DIR INPUT REQUIRED - Default Path:{from_dir}')
        if not baskets:            
            try:
                baskets_in_csv = pd.read_csv(os.path.join(from_dir, 'fundamentals.csv'))['ticker']
            except FileNotFoundError or TypeError:
                logger.normal_logger.info(f'[SCREENER] NO BASKETS EXISTS in {from_dir}')
                return
            baskets = baskets_in_csv ; num_baskets = len(baskets) 
            logger.normal_logger.info(f'[SCREENER] BASKETS INPUT REQUIRED - Default Basket: {num_baskets} baskets in the directory:{from_dir}.')    

        logger.normal_logger.info(f'[SCREENER] UPDATE FOR BASKETS')
        loader = Loader()
        preresults_pdframe = loader.fundamentals_loader(baskets=baskets, from_dir=path, to_dir=path, modules=modules).pdframe.sort_values(by=sort_by)
        results_list =  preresults_pdframe['ticker']
        results_pdframe = preresults_pdframe

        if output=='list':
            return results_list
        if output=='pdframe':
            return results_pdframe

    @staticmethod
    def momentum_screener(baskets=None, from_dir=None, period=None, to_dir=None, output='list'):

        if not period:
            period = 10
            logger.normal_logger.info(f'[SCREENER] PERIOD INPUT REQUIRED - Default Period:{period}')
        if not from_dir:
            from_dir = from_dir
            logger.normal_logger.info(f'[SCREENER] FROM_DIR REQUIRED - Default Path:{from_dir}')
        if not to_dir:
            to_dir = to_dir
            logger.normal_logger.info(f'[SCREENER] TO_DIR INPUT REQUIRED - Default Path:{from_dir}')
        if not baskets:            
            serialized_objects = os.listdir(from_dir)
            serialized_object =list(filter(lambda x: (x[-3:] == 'csv') and ('_' not in x) and ("+" not in x), serialized_objects))
            baskets_in_dir = list(map(lambda x: x[:-4], serialized_object))
            if not baskets_in_dir:
                logger.normal_logger.info(f'[SCREENER] NO BASKETS EXISTS in {from_dir}')
                return
            baskets = baskets_in_dir ; num_baskets = len(baskets) 
            logger.normal_logger.info(f'[SCREENER] BASKETS INPUT REQUIRED - Default Basket: {num_baskets} baskets in the directory:{from_dir}.')    

        logger.normal_logger.info(f'[SCREENER] UPDATE FOR BASKETS')
        loader = Loader()
        loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=to_dir)
        
        logger.normal_logger.info(f'[SCREENER] RECOMMANDATIONS BASED ON LATEST {period} DAYS.')
        
        prllz = parallelize(baskets=baskets, path=from_dir,
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
        
        results_list = list(map(lambda x:mapper[x], np.argsort(recommand)[::-1]))
        results_pdframe = pd.DataFrame(results_list, columns= ['ticker'])
        recent_date = datetime.strftime(prllz.pdframe[prllz.date_column].iloc[-1], "%Y%m%d")
        results_pdframe.to_csv('momentum+screener+{period}+{recent_date}.csv', index=False)
        logger.normal_logger.info('[SCREENER] TOP 10 MOMENTUM FOR {period}: {top10}'.format(period=period, top10=results_list[:10]))
        
        if output=='list':
            return results_list
        if output=='pdframe':
            return results_pdframe

    
