from ..path import refine
from ._core_transfer import DataTransferCore

import os
from typing import Optional, Any, Union, Callable, Iterable
import datetime
from pytz import timezone
import json
from tqdm import tqdm
import pandas as pd
import FinanceDataReader as fdr
from yahooquery import Ticker


def integrated_dataloader(baskets:Iterable[str], path:str=False, source:str='yahooquery')->DataTransferCore:
    if path:
        if path == '.financedatasets':
            pass
        elif loader.firstcall:
            loader.firstcall = False
            loader._initialize(dataset_dirname=refine(path))
    
        if loader.dataset_dirname != refine(path):
            loader._initialize(dataset_dirname=refine(path))
    else:
        loader._initialize()

    with open('.dataset_log.json', 'r') as log:
        download_log = json.loads(json.load(log))
    
    existed_securities = filter(lambda x: x in download_log.keys(), baskets)
    not_existed_securities = filter(lambda x: not x in download_log.keys(), baskets)
    
    # specific asset base loader(1) : yahooquery
    if source == 'yahooquery':
        print('* from yahooquery')
        loader.from_yahooquery(baskets=not_existed_securities, country='united states', progress=True)
        if not bool(loader.failures):
            return loader.from_local(baskets)
        else:
            print('[AILEVER] Download failure list: ', loader.failures)
            return loader.from_local(loader.successes)
    
    # generic loader
    print('* from finance-datareader')
    loader.from_fdr(not_existed_securities)
    if not bool(loader.failures):
        return loader.from_local(baskets)
    else:
        print('[AILEVER] Download failure list: ', loader.failures)
        return loader.from_local(loader.successes)



class Loader:
    def __init__(self):
        self.datacore = DataTransferCore()
        self.firstcall = True
        self.dataset_dirname = '.financedatasets'
        self.log_filename = '.dataset_log.json'
        self.successes = set()
        self.failures = set()
    
    def _initialize(self, dataset_dirname=False):
        today = datetime.datetime.now(timezone('Asia/Seoul'))

        if dataset_dirname:
            self.dataset_dirname = dataset_dirname

        if not os.path.isdir(self.dataset_dirname):
            os.mkdir(self.dataset_dirname)
        if not os.path.isfile(self.log_filename):
            with open(self.log_filename, 'w') as log:
                json.dump(json.dumps(dict(), indent=4), log)
        with open(self.log_filename, 'r') as log:
            download_log = json.loads(json.load(log))

        for existed_security in map(lambda x: x[:-4], filter(lambda x: x[-3:] == 'csv', os.listdir(self.dataset_dirname))):
            download_log[existed_security] = {'WhenDownload':today.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                              'WhenDownload_date':today.strftime('%Y-%m-%d'),
                                              'WhenDownload_Y':today.year,
                                              'WhenDownload_m':today.month,
                                              'WhenDownload_d':today.day, 
                                              'WhenDownload_H':today.hour,
                                              'WhenDownload_M':today.month,
                                              'WhenDownload_S':today.second,
                                              'WhenDownload_TZ':today.tzname(),
                                              'HowDownload':'origin',
                                              'Table_NumRows':None,
                                              'Table_NumColumns':None,
                                              'Table_StartDate':None,
                                              'Table_EndDate':None,
                                              }

        with open(self.log_filename, 'w') as log:
            json.dump(json.dumps(download_log, indent=4), log)

        self.successes.update(download_log.keys())

    def from_local(self, baskets):
        dataset = dict()
        for security in baskets: 
            dataset[security] = pd.read_csv(os.path.join(self.dataset_dirname, f'{security}.csv'))
        self.datacore.dict = dataset
        
        with open(self.log_filename, 'r') as log:
            download_log = json.loads(json.load(log))
        self.datacore.log = download_log
        return self.datacore

    def from_yahooquery(self, baskets, asynchronouse=False, backoff_factor=0.3, country='united states',
                        formatted=False, max_workers=8, proxies=None, retry=5, status_forcelist=[404, 429, 500, 502, 503, 504], timeout=5,
                        validate=False, verify=True, progress=True):
        baskets = list(baskets)
        successes = dict()
        _successes = list()
        failures = list()
        try:
            ticker = Ticker(symbols=baskets, asynchronouse=asynchronouse, backoff_factor=backoff_factor, country=country,
                            formatted=formatted, max_workers=max_workers, proxies=proxies, retry=retry, status_forcelist=status_forcelist, timeout=timeout,
                            validate=validate, verify=verify, progress=progress)
            securities = ticker.history(period="max", interval="1d", start=None, end=None, adj_timezone=True, adj_ohlc=True)
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
                security_frame.to_csv(os.path.join(self.dataset_dirname, f'{security}.csv'))
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
                security_frame.to_csv(os.path.join(self.dataset_dirname, f'{security}.csv'))
                successes[security] = {'Table_NumRows':security_frame.shape[0],
                                       'Table_NumColumns':security_frame.shape[1],
                                       'Table_StartDate':security_frame.index[0].strftime('%Y-%m-%d'),
                                       'Table_EndDate':security_frame.index[-1].strftime('%Y-%m-%d'),
                                       }

        self.successes.update(_successes)
        self.failures.update(failures)
        self._logger_for_successes(message='from_yahooquery', updated_basket_info=successes)

    def from_fdr(self, baskets):
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
        self._logger_for_successes(message='from_fdr', updated_basket_info=successes)
        
    def _logger_for_successes(self, message, updated_basket_info):
        today = datetime.datetime.now(timezone('Asia/Seoul'))

        with open(self.log_filename, 'r') as log:
            download_log = json.loads(json.load(log))
        
        updated_basket = updated_basket_info.keys()
        for security in updated_basket:
            download_log[security] = {'WhenDownload':today.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                      'WhenDownload_date':today.strftime('%Y-%m-%d'),
                                      'WhenDownload_Y':today.year,
                                      'WhenDownload_m':today.month,
                                      'WhenDownload_d':today.day, 
                                      'WhenDownload_H':today.hour,
                                      'WhenDownload_M':today.month,
                                      'WhenDownload_S':today.second,
                                      'WhenDownload_TZ':today.tzname(),
                                      'HowDownload':message,
                                      'Table_NumRows':updated_basket_info[security]['Table_NumRows'],
                                      'Table_NumColumns':updated_basket_info[security]['Table_NumColumns'],
                                      'Table_StartDate':updated_basket_info[security]['Table_StartDate'],
                                      'Table_EndDate':updated_basket_info[security]['Table_EndDate'],
                                      }

        with open(self.log_filename, 'w') as log:
            json.dump(json.dumps(download_log, indent=4), log)



loader = Loader()


