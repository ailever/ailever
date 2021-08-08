from ..path import refind
from ._fmlops_policy import fmlops_bs
from ._base_transfer import DataTransferCore
from .logger import update_log



import os
from typing import Optional, Any, Union, Callable, Iterable
import datetime
from pytz import timezone
import json
from tqdm import tqdm
import pandas as pd
import FinanceDataReader as fdr
from yahooquery import Ticker

r"""
Integrated Loader for all types of financial datasets
Source: yahooquery, financial datareader


Unites States Stock market Timezone : EST 09:30 ~ 16:00
"""



base_dir = dict()
base_dir['root'] = fmlops_bs.local_system.root
base_dir['rawdata_repository'] = fmlops_bs.local_system.rawdata_repository
base_dir['metadata_store'] = fmlops_bs.local_system.metadata_store
base_dir['feature_store'] = fmlops_bs.local_system.feature_store
base_dir['model_registry'] = fmlops_bs.local_system.model_registry
base_dir['source_repotitory'] = fmlops_bs.local_system.source_repository



dataset_dirname = os.path.join(base_dir['root'], base_dir['rawdata_repository'])
log_dirname = os.path.join(base_dir['root'], base_dir['metadata_store'])
update_log_file = update_log


class Loader() -> Datacore:

    baskets = None
    from_dir = None
    to_dir = None
    datacore = None

    def __init__(self, baskets):

        self.baskets = baskets
        self.datacore = DataTransferCore()
        self.from_dir = dataset_dirname 
        self.to_dir =  dataset_dirname

    def ohlcv_loader(self, baskets:Iterable[str], from_dir=dataset_dirname, to_dir=dataset_dirname, 
                    update_log_dir=log_dirname, update_log_file=update_log_file['ohlvc'], country='united states', frequency='1d', source='yahooquery'):

        r"""---------- Initialzing dataset directories ----------"""
        
        if not from_dir:
            from_dir = self.from_dir 
        if not to_dir:
            to_dir = self.to_dir
                
        r"""---------- Initialzing Timezone ----------""" 

        if country = 'united states':
            today = datetime.datetime.now(timezone('US/Eastern'))
        if country = 'Korea':
            today = datetime.datetime.now(timezone('Asia/Seoul'))

        r"""--------- Initializing UPDATE log directoreis ----------"""

        if update_dir:
            if not os.path.isdir(update_log_dir):
                os.mkdir(update_log_dir)
        if not update_dir:
            if not os.path.isdir(update_log_dir)
                os.mkdir(update_log_dir)
        
        r"""---------- Initializing UPDATE log file ----------"""

        if not os.path.isfile(os.path.join(update_log_dirname, update_log_file)):
            with open(os.path.join(update_log_dirname, update_log_file), 'w') as log:
                json.dump(json.dumps(dict(), indent=4), log)
            download_log = dict()            
            for existed_security in map(lambda x: x[:-4], filter(lambda x: x[-3:] == 'csv', os.listdir(from_dir))):
                download_log[existed_security] = {'WhenDownload':today.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                                'WhenDownload_TZ':today.tzname(),
                                                'HowDownload':'origin',
                                                'Table_NumRows':None,
                                                'Table_NumColumns':None,
                                                'Table_StartDate':None,
                                                'Table_EndDate':None,
                                                }
                
            with open(os.path.join(update_log_dirname, update_log_file), 'w') as log:
                json.dump(json.dumps(download_log, indent=4), log)
            
        r"---------- Initializing SELECT baskets ----------"
        
        ## Update log file loading
            
        with open(os.path.join(update_log_dirname, update_log_file), 'r') as log:
            update_log = json.loads(json.load(log))
            

        ### Case 1) No baskets -> SELECT baskets are all the tickers in datset_dir
        
        if not baskets:
            
            serialized_objects = os.listdir(from_dir)
            serialized_object =list(filter(lambda x: x[-3:] == 'csv', serialized_objects))
            baskets_in_dir = list(map(lambda x: x[:-4], serialized_objects))
            
            in_the_baskets = list(map(update_log.keys().get, baskets_in_dir))
            dates = [value["Table_EndDate"] for value in in_the_baskets]
            
            format_time = '%Y-%m-%d'
            if datetime.now(timezone('US/Eastern')) > max(list(map(lambda x: datetime.strptime(x, format_time), dates))):
                select_baskets = in_the_baskets


        ### Case 2) 

        in_the_baskets = list(map(update_log.keys().get, baskets_in_dir))
        dates = [value["Table_EndDate"] for value in in_the_baskets]
            
        ### Case 2-1) Baskets are not in the log before -> SELECT baskets are all the tickers in the bakset
        
        if not baskets in list(update_log.key()):
            select_baskets =  baskets

        ### Case 2-2) When no Table end date are recorded (eg. when log file was newly made with in-place outsourced csv files)
        
        if None in in_basket_dates:
            select_baskets = baskets
        
        ### Case 2-3) all tickers in basket was in exisitng logger but they are outdated
        
        format_time = '%Y-%m-%d'
        if datetime.now(timezone('US/Eastern')) > max(list(map(lambda x: datetime.strptime(x, format_time), in_basket_dates))):
            select_baskets = baskets     
        
        r""" ---------- Executing DataVendor ----------"""    
        
        datavendor = DataVendor(baskets=select_baskets, country=country)
        
        r""" --------- ohlcv from yahooquery ----------"""

        if interval == '1d':
            print('TBD * from yahooquery currently not supporting. Please make sure dataset directories are seperate by frequency')

        if source == 'yahooquery':
            print('* from yahooquery')
            datavendor.ohlcv_from_yahooquery(baskets=baskets, from_dir=from_dir, to_dir=to_dir, 
                                            update_log_dir=update_log_dir, update_log_file=update_log_file, interval=interval, country=country, progress=True,)
            
            if not bool(datavendor.failures):
                datavendor.ohlcv_from_local(baskets=baskets, from_dir=from_dir, to_dir=to_dir, 
                        update_log_dir=update_log_dir, update_log_file=update_log_file, interval=interval, country=country)

            else:
                print('[AILEVER] Download failure list: ', datavendor.failures)
                datavendor.ohlcv_from_local(baskets=baskets, from_dir=from_dir, to_dir=to_dir, 
                        update_log_dir=update_log_dir, update_log_file=update_log_file, interval=interval, country=country)
            
            self.datacore.dict = datavendor.dict
            self.datacore.log = datavendor.log
            return


        r""" ---------- ohlcv from fdr reader ----------"""

        print('* from finance-datareader')
        datavendor.from_fdr(baskets=baskets, from_dir=from_dir, to_dir=to_dir, 
            update_log_dir=update_log_dir, update_log_file=update_log_file, interval=interval, country=country))
        if not bool(datavendor.failures):
            datavendor.ohlcv_from_local(baskets=baskets, from_dir=from_dir, to_dir=to_dir, 
                    update_log_dir=update_log_dir, update_log_file=update_log_file, interval=None, country=None))
        else:
            print('[AILEVER] Download failure list: ', datavendor.failures)
            datavendor.ohlcv_from_local(loader.successes)
        
        self.datacore.dict = datavendor.dict
        self.datacore.log = datavendor.log

               

    def fundamentals_loader(self, baskets:Iterable[str], from_dir=dataset_dirname, to_dir=dataset_dirname, 
                    update_log_dir=log_dirname, update_log_file=None, country='united states', modules="all_modules" ,source='yahooquery'):

        r"""---------- Initialzing dataset directories ----------"""
        
        if not from_dir:
            from_dir = self.from_dir 
        if not to_dir:
            to_dir = self.to_dir
                
        r"""---------- Initialzing Timezone ----------""" 

        if country = 'united states':
            today = datetime.datetime.now(timezone('US/Eastern'))
        if country = 'Korea':
            today = datetime.datetime.now(timezone('Asia/Seoul'))


        datavendor = DataVendor(baskets=select_baskets, country=country)
        
        r""" --------- fundamentals from yahooquery ----------"""


        if source == 'yahooquery':
            print('* from yahooquery')
            datavendor.fundamentals_from_yahooquery(baskets=baskets, from_dir=from_dir, to_dir=to_dir, 
                                            update_log_dir=update_log_dir, update_log_file=update_log_file, country=country, modules=modules,progress=True)
            
            if not bool(datavendor.failures):
                datavendor.ohlcv_from_local(baskets=baskets, from_dir=from_dir, to_dir=to_dir, 
                        update_log_dir=update_log_dir, update_log_file=update_log_file, country=country, modules=modules, progress=True)




r"""DataVendor Crawling Class - Return cls.dict and cls.log will be passed onto Loader.datacore frame"""       

class DataVendor():

    def __init__(self, baskets=None, country=None)
        
        self.log = None
        self.dict = None
        self.successes = set()
        self.failures = set()
        
        self.baskets = baskets
        self.country = country
    
    r"""OHLCV raw data download process
        
        |--- Directly downloaded from yahooquery or fdr qudry or any other data vendor(TBD)
        |--- Download to LOCAL DIRECTORIES (to_dir) through ohlcv_from_[datavendor]
        |--- Load from LOCAL Directories (from_dir) through ohlcv_from_local() 
    
    """

    def ohlcv_from_local(self, baskets=None, from_dir=None, to_dir=None, update_log_dir=None, update_log_file=None, interval=None, country=None):
        
        r"""Initializing Args"""

        if not baskets:
            baskets = self.baskets
        if not country:
            coutry = self.country
        
        from_dir = from_dir
        update_log_dir = update_log_dir
        update_log_file = update_log_file
        
        r"""Load from LOCAL directories"""

        dataset = dict()
        for security in baskets: 
            dataset[security] = pd.read_csv(os.path.join(from_dir, f'{security}.csv'))
        self.dict = dataset
        
        with open(os.path.join(self.update_log_dirname, self.update_log_file), 'r') as log:
            update_log = json.loads(json.load(log))

        self.log = update_log
        return self

    def ohlcv_from_yahooquery(self, baskets=None, from_dir=None, to_dir=None, update_log_dir=None, update_log_file=None, interval=None, country=None, progress=True,
                        asynchronouse=False, backoff_factor=0.3, formatted=False, max_workers=12, proxies=None, retry=5, 
                        status_forcelist=[404, 429, 500, 502, 503, 504], timeout=5, validate=False,verify=True)
        
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
            be_in_memory = set(map(lambda x:x[0], securities.index))
            _successes.extend(be_in_memory)
            failures.extend(filter(lambda x: not x in be_in_memory, baskets))
            for security in be_in_memory:
                security_frame = securities.loc[security]
                security_frame.to_csv(os.path.join(to_dir, f'{security}.csv'))
                successes[security] = {'Table_NumRows':security_frame.shape[0],
                                       'Table_NumColumns':security_frame.shape[1],
                                       'Table_StartDate':security_frame.index[0].strftime('%Y-%m-%d'),
                                       'Table_EndDate':security_frame.index[-1].strftime('%Y-%m-%d'),
                                       }
        
        elif isinstance(securities, dict):
            be_in_memory = map(lambda x:x[0], filter(lambda x:not isinstance(x[1], str), zip(securities.keys(), securities.values())))
            not_in_memory = map(lambda x:x[0], filter(lambda x:isinstance(x[1], str), zip(securities.keys(), securities.values())))
            _successes.extend(be_in_memory)
            failures.extend(not_in_memory)
            for security in _successes:
                security_frame = fdr.DataReader(security)
                security_frame.to_csv(os.path.join(to_dir, f'{security}.csv'))
                successes[security] = {'Table_NumRows':security_frame.shape[0],
                                       'Table_NumColumns':security_frame.shape[1],
                                       'Table_StartDate':security_frame.index[0].strftime('%Y-%m-%d'),
                                       'Table_EndDate':security_frame.index[-1].strftime('%Y-%m-%d'),
                                       }


        self.successes.update(_successes)
        self.failures.update(failures)
        self._logger_for_successes(message='from_yahooquery', updated_basket_info=successes, 
                                    update_log_dir=update_log_dir, update_log_file=update_log_file, country=country)
        

    def ohlcv_from_fdr(self, baskets=None, from_dir=None, to_dir=None, update_log_dir=None, update_log_file=None, interval=None, country=None):
        
        if not baskets:
            baskets = self.baskets
        if not country:
            coutry = self.country

        successes = dict()
        failures = list()
        for security in tqdm(baskets):
            try:
                security_frame = fdr.DataReader(security)
                security_frame.to_csv(os.path.join(self.dataset_dirname, f'{security}.csv'))
                successes[security] = {'Table_NumRows':security_frame.shape[0],
                                       'Table_NumColumns':security_frame.shape[1],
                                       'Table_StartDate':security_frame['Date'].iloc[0],
                                       'Table_EndDate':security_frame['Date'].iloc[-1],
                                       }
            except:
                failures.append(security)
                continue

        self.successes.update(successes.keys())
        for success in list(filter(lambda x: x in self.successes, self.failures)):
            self.failures.remove(success)
        self.failures.update(failures)
        self._logger_for_successes(message='from_fdr', updated_basket_info=successes, 
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
        
        if country = 'united states':
            today = datetime.datetime.now(timezone('US/Eastern'))
        if country = 'korea':
            today = datetime.datetime.now(timezone('Asia/Seoul'))

        with open(os.path.join(update_log_dir, update_log_file), 'r') as log:
            update_log = json.loads(json.load(log))
        
        updated_basket = updated_basket_info.keys()
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

   
        
        
        






