from ...path import refine
from .._fmlops_policy import fmlops_bs
from .._base_transfer import DataTransferCore
from ..logger import update_log
from ..logger import Logger

from typing import Optional, Any, Union, Callable, Iterable
from pytz import timezone
from tqdm import tqdm
from yahooquery import Ticker

import pandas as pd
import FinanceDataReader as fdr
import json

import os
import datetime
import pandas as pd

"""DataVendor Crawling Class - Return cls.dict and cls.log will be passed onto Loader.datacore frame"""       

base_dir = dict()
base_dir['root'] = fmlops_bs.local_system.root
base_dir['rawdata_repository'] = fmlops_bs.local_system.rawdata_repository
base_dir['metadata_store'] = fmlops_bs.local_system.metadata_store
base_dir['feature_store'] = fmlops_bs.local_system.feature_store
base_dir['model_registry'] = fmlops_bs.local_system.model_registry
base_dir['source_repotitory'] = fmlops_bs.local_system.source_repository

logger = Logger()
dataset_dirname = os.path.join(base_dir['root'], base_dir['rawdata_repository'])

class DataVendor(DataTransferCore):

    def __init__(self, baskets=None, country=None):
        
        self.successes = dict()
        self.failures = dict()
        self.baskets = baskets
        self.country = country
    
        r"""OHLCV raw data download process
        
        |--- Directly downloaded from yahooquery or fdr qudry or any other data vendor(TBD)
        |--- Download to LOCAL DIRECTORIES (to_dir) through ohlcv_from_[datavendor]
        |--- Load from LOCAL Directories (from_dir) through ohlcv_from_local() 
    
        """

    def ohlcv_from_local(self, baskets=None, from_dir=None, to_dir=None, update_log_dir=None, update_log_file=None):
        
        r"""Initializing Args"""
        if not baskets:
            baskets = self.baskets
        
        from_dir = from_dir
        update_log_dir = update_log_dir
        update_log_file = update_log_file
        
        r"""Load from LOCAL directories"""
        dataset = dict()
        for security in baskets: 
            dataset[security] = pd.read_csv(os.path.join(from_dir, f'{security}.csv'))
        self.dict = dataset
        
        with open(os.path.join(update_log_dir, update_log_file), 'r') as log:
            update_log = json.loads(json.load(log))

        self.log = update_log
        return self

    def ohlcv_from_yahooquery(self, baskets=None, from_dir=None, to_dir=None, update_log_dir=None, update_log_file=None, interval=None, country=None, progress=True,
                        asynchronouse=False, backoff_factor=0.3, formatted=False, max_workers=12, proxies=None, retry=5, 
                        status_forcelist=[404, 429, 500, 502, 503, 504], timeout=5, validate=False,verify=True):
        
        r"""Initializing Args"""
        if not baskets:
            baskets = self.baskets
        if not country:
            coutry = self.country
        
        baskets = list(baskets)
        to_dir = to_dir
        update_log_dir = update_log_dir
        update_log_file = update_log_file

        successes = dict()
        _successes = list()
        failures = list()
        
        r"""Request from [Data Vendor]"""
        try:
            ticker = Ticker(symbols=baskets, asynchronouse=asynchronouse, backoff_factor=backoff_factor, country=country,
                            formatted=formatted, max_workers=max_workers, proxies=proxies, retry=retry, status_forcelist=status_forcelist, timeout=timeout,
                            validate=validate, verify=verify, progress=progress)
            securities = ticker.history(period="max", interval=interval, start=None, end=None, adj_timezone=True, adj_ohlc=True)
    
        except:
            failures.extend(baskets)
            self.failures.update(failures)
            return
        

        if isinstance(securities, pd.core.frame.DataFrame):
            
            r"""ohlcv pre-processing"""
            securities.columns = list(map(lambda x: x.lower(), securities.columns))
            securities.index.names = list(map(lambda x: x.lower(), securities.index.names))
            securities = securities[['open', 'high', 'low', 'close', 'volume']]
            
            be_in_memory = set(map(lambda x:x[0], securities.index))
            _successes.extend(be_in_memory)
            failures.extend(filter(lambda x: not x in be_in_memory, baskets))
            
            for security in be_in_memory:
                security_frame = securities.loc[security]
                
                print(security_frame)
                security_frame.to_csv(os.path.join(to_dir, f'{security}.csv'))
                successes[security] = {'Table_NumRows':security_frame.shape[0],
                                       'Table_NumColumns':security_frame.shape[1],
                                       'Table_StartDate':security_frame.index[0].strftime('%Y-%m-%d'),
                                       'Table_EndDate':security_frame.index[-1].strftime('%Y-%m-%d'),
                                       }
            print(successes)
        elif isinstance(securities, dict):
            be_in_memory = map(lambda x:x[0], filter(lambda x:not isinstance(x[1], str), zip(securities.keys(), securities.values())))
            not_in_memory = map(lambda x:x[0], filter(lambda x:isinstance(x[1], str), zip(securities.keys(), securities.values())))
            
            _successes.extend(be_in_memory)
            failures.extend(not_in_memory)

            for security in be_in_memoriy:
                security_frame = securities.loc[security]
                
                r"""ohlcv pre-processing"""
                security_frame.columns = list(map(lambda x: x.lower(), security_frame.columns))
                security_frame.index.names = ['date']
                security_frame = security_frame[['open', 'high', 'low', 'close', 'volume']]

                security_frame.to_csv(os.path.join(to_dir, f'{security}.csv'))
                successes[security] = {'Table_NumRows':security_frame.shape[0],
                                       'Table_NumColumns':security_frame.shape[1],
                                       'Table_StartDate':security_frame.index[0].strftime('%Y-%m-%d'),
                                       'Table_EndDate':security_frame.index[-1].strftime('%Y-%m-%d'),
                                       }


        self.successes.update(successes)
        self.failures.update(failures)
        self._logger_for_successes(message='from_yahooquery', updated_basket_info=self.successes, 
                                    update_log_dir=update_log_dir, update_log_file=update_log_file, country=country)
        

    def ohlcv_from_fdr(self, baskets=None, from_dir=None, to_dir=None, update_log_dir=None, update_log_file=None, interval=None, country=None):
        if not baskets:
            baskets = self.baskets
        if not country:
            coutry = self.country
        
        baskets = list(baskets)
        to_dir = to_dir
        update_log_dir = update_log_dir
        update_log_file = update_log_file

        successes = dict()
        failures = list()
        for security in tqdm(baskets):
            try:
                security_frame = fdr.DataReader(security)
                
                r"""ohlcv pre-processing"""
                security_frame.columns = list(map(lambda x: x.lower(), security_frame.columns)) 
                security_frame.index.names = list(map(lambda x: x.lower(), security_frame.index.names))
                security_frame = security_frame[['open', 'high', 'low', 'close', 'volume']]

                security_frame.to_csv(os.path.join(to_dir, f'{security}.csv'))
                successes[security] = {'Table_NumRows':security_frame.shape[0],
                                       'Table_NumColumns':security_frame.shape[1],
                                       'Table_StartDate':security_frame['Date'].iloc[0],
                                       'Table_EndDate':security_frame['Date'].iloc[-1],
                                       }
            except:
                failures.append(security)
                continue
        
        self.successes.update(successes)
        for success in list(filter(lambda x: x in self.successes, self.failures)):
            self.failures.remove(success)
        self.failures.update(failures)
        self._logger_for_successes(message='from_fdr', updated_basket_info=self.successes, 
                                    update_log_dir=update_log_dir, update_log_file=update_log_file, country=country)
    


    def fundamentals_from_yahooquery(self, baskets=None, from_dir=None, to_dir=None, update_log_dir=None, update_log_file=None, country=None, modules="all_modules",process=True,        
                                    asynchronouse=False, backoff_factor=0.3, formatted=False, max_workers=12, proxies=None, retry=5, 
                                    status_forcelist=[404, 429, 500, 502, 503, 504], timeout=5, validate=False,verify=True):

        r""" No Update log formatted in json is currently available"""
        if not baskets:
            baskets = self.baskets
        if not country:
            coutry = self.country

        successes = dict()
        failures = list()
        
        try:
            ticker = Ticker(symbols=baskets, asynchronouse=asynchronouse, backoff_factor=backoff_factor, country=country,
                        formatted=formatted, max_workers=max_workers, proxies=proxies, retry=retry, status_forcelist=status_forcelist, timeout=timeout,
                        validate=validate, verify=verify, progress=progress)
            
            if modules == "all_modules":

                securities = ticker.all_modules
                secutities = self.dict
                return
            
            r"""
            modules = ['assetProfile', 'earnings']
            ticker = get_modules(modules)

            [
            'assetProfile', 'recommendationTrend', 'cashflowStatementHistory',
            'indexTrend', 'defaultKeyStatistics', 'industryTrend', 'quoteType',
            'incomeStatementHistory', 'fundOwnership', 'summaryDetail', 'insiderHolders',
            'calendarEvents', 'upgradeDowngradeHistory', 'price', 'balanceSheetHistory',
            'earningsTrend', 'secFilings', 'institutionOwnership', 'majorHoldersBreakdown',
            'balanceSheetHistoryQuarterly', 'earningsHistory', 'esgScores', 'summaryProfile',
            'netSharePurchaseActivity', 'insiderTransactions', 'sectorTrend',
            'incomeStatementHistoryQuarterly', 'cashflowStatementHistoryQuarterly', 'earnings',
            'pageViews', 'financialData'
            ]
            """

        except:
            failures.extend(baskets)
            self.failures.update(failures)
            return
 


    """UPDATE log formatted in json ---> Currently only avilable for ohlcv datasets"""

    def _logger_for_successes(self, message=False, updated_basket_info=False, 
                                update_log_dir=None, update_log_file=None, country=False):
        
        if country == 'united states':
            today = datetime.datetime.now(timezone('US/Eastern'))
        if country == 'korea':
            today = datetime.datetime.now(timezone('Asia/Seoul'))

        with open(os.path.join(update_log_dir, update_log_file), 'r') as log:
            update_log = json.loads(json.load(log))
        
        updated_basket = list(updated_basket_info.keys())
        for security in updated_basket:
            update_log[security] = {'WhenDownload':today.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                     'WhenDownload_TZ':today.tzname(),
                                      'HowDownload':message,
                                      'Table_NumRows':updated_basket_info[security]['Table_NumRows'],
                                      'Table_NumColumns':updated_basket_info[security]['Table_NumColumns'],
                                      'Table_StartDate':updated_basket_info[security]['Table_StartDate'],
                                      'Table_EndDate':updated_basket_info[security]['Table_EndDate'],
                                      }

        with open(os.path.join(update_log_dir, update_log_file), 'w') as log:
            json.dump(json.dumps(update_log, indent=4), log)
        logger.normal_logger.info(f'{updated_basket} updates are logged in {update_log_file}')    

   
