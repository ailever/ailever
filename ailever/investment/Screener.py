from re import I
from ailever.investment import __fmlops_bs__ as fmlops_bs
from ._base_transfer import DataTransferCore
from .logger import Logger
from .fmlops_loader_system import parallelize
from .fmlops_loader_system import Loader
from .fmlops_loader_system import Preprocessor
from .fmlops_loader_system.DataVendor import DataVendor
from pytz import timezone

import csv
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
    fundamentals_modules_fromyahooquery_dict = DataVendor.fundamentals_modules_fromyahooquery_dict
    fundamentals_modules_fromyahooquery = DataVendor.fundamentals_modules_fromyahooquery
    fmf = DataVendor.fundamentals_modules_fromyahooquery
    
    def __init__(self):
        self._decision_profiling()

    def _decision_profiling(self):
        self.decision_matrix  = None


    @staticmethod
    def fundamentals_screener(baskets=None, from_dir=None, to_dir=None, period=None, modules=None, sort_by=None, drop_negative=True, interval=None, country='united states', output='list'):
        """
        sory_by option
        ['DividendYield', 'FiveYrsDividendYield', 'DividendRate', 'Beta', 'EVtoEBITDA', 'Marketcap', 'EnterpriseValue']"""
        module_dict = Screener.fundamentals_modules_fromyahooquery_dict
        order_type = {'DividendYield': True, 'FiveYrsDiviendYield': False, 'DiviendRate': False, 'Beta': True, 'EVtoEBITDA': True, 'Marketcap': False, 'EnterpriseValue': False}
        if not from_dir:
            from_dir = from_dir   
            logger.normal_logger.info(f'[SCREENER] FROM_DIR INPUT REQUIRED - Default Path:{from_dir}')
        if not to_dir:
            to_dir = to_dir
            logger.normal_logger.info(f'[SCREENER] TO_DIR INPUT REQUIRED - Default Path:{to_dir}')
        if country == 'united states':
            today = datetime.now(timezone('US/Eastern'))
            tz = timezone('US/Eastern')
        if country == 'korea':
            today = datetime.now(timezone('Asia/Seoul'))
        if not interval:
            interval = '1d'
        if not baskets: 
            if not os.path.isfile(os.path.join(from_dir, 'fundamentals.csv')):
                baskets_in_csv = list()
                logger.normal_logger.info(f"[SCREENER] NO BASKETS EXISTS from {from_dir}")
                return
            if not os.path.isfile(os.path.join(from_dir, 'fundamentals.csv')):
                baskets_in_csv = pd.read_csv(os.path.join(from_dir, 'fundamentals.csv'))['ticker'].tolist()         
            baskets = baskets_in_csv ; num_baskets = len(baskets) 
            logger.normal_logger.info(f'[SCREENER] BASKETS INPUT REQUIRED - Default Basket: {num_baskets} baskets in the directory:{from_dir}.')    

        logger.normal_logger.info(f'[SCREENER] UPDATE FOR BASKETS')
        loader = Loader()
        if not modules:
            modules = list(loader.fmf)
        preresults_pdframe = loader.fundamentals_loader(baskets=baskets, from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, modules=modules).pdframe[[module_dict[sort_by][2]]].sort_values(module_dict[sort_by][2], ascending=order_type[sort_by])
        if drop_negative:
            preresults_pdframe = preresults_pdframe[preresults_pdframe>0]
            logger.normal_logger.info('[SCEENER] DROP NEGATIVE VALUES FOR RANKING')
        results_list =  preresults_pdframe.index.tolist() 
        top10 = results_list[:10]
        results_pdframe = preresults_pdframe
        logger.normal_logger.info(f'[SCREENER] {sort_by} RANK YIELED: TOP 10 {top10}')
        if output=='list':
            return results_list
        if output=='pdframe':
            return results_pdframe

    @staticmethod
    def momentum_screener(baskets=None, from_dir=None, interval=None, country='united stated', period=None, to_dir=None, output='list'):
        if not period:
            period = 10
            logger.normal_logger.info(f'[SCREENER] PERIOD INPUT REQUIRED - Default Period:{period}')
        if not from_dir:
            from_dir = from_dir
            logger.normal_logger.info(f'[SCREENER] FROM_DIR REQUIRED - Default Path:{from_dir}')
        if not to_dir:
            to_dir = to_dir
            logger.normal_logger.info(f'[SCREENER] TO_DIR INPUT REQUIRED - Default Path:{to_dir}')
        if not interval:
            interval ='1d'
            logger.normal_logger.info(f'[SCREENER] DEFAULT INTERVAL {interval}')
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
        loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=to_dir, interval=interval, country=country)
        
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
        results_pdframe.to_csv(f'momentum+screener+{period}+{recent_date}.csv', index=False)
        logger.normal_logger.info('[SCREENER] TOP 10 MOMENTUM FOR {period}: {top10}'.format(period=period, top10=results_list[:10]))
        
        if output=='list':
            return results_list
        if output=='pdframe':
            return results_pdframe
    
    @staticmethod
    def pct_change_screener(baskets=None, from_dir=None, to_dir=None, window=None, sort_by=None, ascending=None, interval=None, country='united states', output='list'):
        if not from_dir:
            path = from_dir   
            logger.normal_logger.info(f'[SCREENER] FROM_DIR INPUT REQUIRED - Default Path:{path}')
        if not to_dir:
            to_dir = to_dir
            logger.normal_logger.info(f'[SCREENER] TO_DIR INPUT REQUIRED - Default Path:{from_dir}')
        if country == 'united states':
            today = datetime.now(timezone('US/Eastern'))
            tz = timezone('US/Eastern')
        if country == 'korea':
            today = datetime.now(timezone('Asia/Seoul'))
        if not interval:
            interval = '1d'
        if not window:
            window = [1,5,20,60,120,240]
        if not sort_by:
            sort_by = 1
        sort_by_column = f'close+change{sort_by}'
        if not ascending:
            ascending = False
        if not baskets: 
            if not os.path.isfile(os.path.join(from_dir, 'pct_change.csv')):
                baskets_in_csv = list()
                logger.normal_logger.info(f"[SCREENER] NO BASKETS EXISTS from {from_dir}")
                return
            if not os.path.isfile(os.path.join(from_dir, 'fundamentals.csv')):
                baskets_in_csv = pd.read_csv(os.path.join(from_dir, 'pct_change.csv'))['ticker'].tolist()         
            baskets = baskets_in_csv ; num_baskets = len(baskets) 
            logger.normal_logger.info(f'[SCREENER] BASKETS INPUT REQUIRED - Default Basket: {num_baskets} baskets in the directory:{from_dir}.')    

        logger.normal_logger.info(f'[SCREENER] ACCESS PREPROCESSOR')
        pre = Preprocessor()
        preresults_dict = pre.pct_change(baskets=baskets, from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, target_column='close', window=window, merge=False, kind='ticker').dict
        main_frame_list = list()
        for ticker in list(preresults_dict.keys()):
            ticker_frame = preresults_dict[ticker].iloc[-1:]
            ticker_frame['ticker'] = ticker
            ticker_frame.reset_index(inplace=True)
            ticker_frame.set_index('ticker', inplace=True)
            main_frame_list.append(ticker_frame)
        main_pdframe = pd.concat(main_frame_list, axis=0)
        
        try:
            main_pdframe = main_pdframe.sort_values(sort_by_column, ascending=ascending)
        except:
            logger.normal_logger.info(f"[SCREENER] NO RESPECTIVE PCT CHANGE EXISTS: Winodw {window}")
        
        results_pdframe = main_pdframe
        results_list = main_pdframe.index.tolist()
        top10 = results_list[:10]
        logger.normal_logger.info(f'[SCREENER] {sort_by_column} RANK YIELED: TOP 10 {top10}')
        if output=='list':
            return results_list
        if output=='pdframe':
            return results_pdframe

    @staticmethod
    def momentum_screener(baskets=None, from_dir=None, interval=None, country='united states', period=None, to_dir=None, output='list'):
        if not period:
            period = 10
            logger.normal_logger.info(f'[SCREENER] PERIOD INPUT REQUIRED - Default Period:{period}')
        if not from_dir:
            from_dir = from_dir
            logger.normal_logger.info(f'[SCREENER] FROM_DIR REQUIRED - Default Path:{from_dir}')
        if not to_dir:
            to_dir = to_dir
            logger.normal_logger.info(f'[SCREENER] TO_DIR INPUT REQUIRED - Default Path:{from_dir}')
        if not interval:
            interval ='1d'
            logger.normal_logger.info(f'[SCREENER] DEFAULT INTERVAL {interval}')
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
        loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=to_dir, interval=interval, country=country)
        
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
        results_pdframe.to_csv(f'momentum+screener+{period}+{recent_date}.csv', index=False)
        logger.normal_logger.info('[SCREENER] TOP 10 MOMENTUM FOR {period}: {top10}'.format(period=period, top10=results_list[:10]))
        
        if output=='list':
            return results_list
        if output=='pdframe':
            return results_pdframe    


    @staticmethod
    def csv_compiler(from_dir, to_dir, now, format_time, target_list):
        return
        csv_list = list()
        with open(from_dir, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                csv_list.append(row)

        "recent_record = tz.localize(datetime.strptime(csv_list[-1][0], format_time))"
        if now < recent_record:
            logger.normal_logger.info("[SCREENER] File IS UP-TO-DATE")
        if now >= recent_record:
            with open(to_dir, 'a', newline="") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(target_list.insert(0, datetime.strftime(now, format_time)))
                logger.normal_logger.info('[SCREEENR] {now} LIST ADDED TO {to_dir}')
