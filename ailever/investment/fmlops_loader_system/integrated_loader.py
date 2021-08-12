from ailever.investment import __fmlops_bs__ as fmlops_bs
from numpy import exp
from pandas.core import indexing

from ...path import refine
from .._base_transfer import DataTransferCore
from ..logger import update_log
from ..logger import Logger
from .DataVendor import DataVendor

from typing import Optional, Any, Union, Callable, Iterable
from pytz import timezone
from yahooquery import Ticker

import datetime
import pandas as pd
import os
import json

r"""
Integrated Loader for all types of financial datasets
DataVendor Source: yahooquery, financial datareader

Unites States Stock market Timezone : EST 09:30 ~ 16:00
"""

base_dir = dict()
base_dir['root'] = fmlops_bs.local_system.root.name
base_dir['metadata_store'] = fmlops_bs.local_system.root.metadata_store.name
base_dir['feature_store'] = fmlops_bs.local_system.root.feature_store.name
base_dir['model_registry'] = fmlops_bs.local_system.root.model_registry.name
base_dir['source_repotitory'] = fmlops_bs.local_system.root.source_repository.name

logger = Logger()
dataset_dirname = os.path.join(base_dir['root'], base_dir['feature_store'])
log_dirname = os.path.join(base_dir['root'], base_dir['metadata_store'])
update_log_dict = update_log


class Loader():

    baskets = None
    from_dir = None
    to_dir = None
    fundamentals_modules_fromyahooquery = DataVendor.fundamentals_modules_fromyahooquery
    fmf = DataVendor.fundamentals_modules_fromyahooquery


    def __init__(self, baskets=None, from_dir=dataset_dirname, to_dir=dataset_dirname, update_log_dir=log_dirname, update_log_file=None):

        self.baskets = baskets
        self.from_dir = from_dir
        self.to_dir = to_dir
        self.update_log_dict = update_log_dict
        self.update_log_dir = update_log_dir
        self.update_log_file = update_log_file
        

    def ohlcv_loader(self, baskets:Iterable[str]=None, from_dir=None, to_dir=None, 
                    update_log_dir=None, update_log_file=None, country='united states', interval='1d', source='yahooquery'):

        r"""---------- Initialzing dataset directories ----------"""
        if not from_dir:
            from_dir = self.from_dir    
            logger.normal_logger.info(f'[LOADER] FROM_DIR INPUT REQUIRED - Default Path:{from_dir}')
        if not to_dir:
            to_dir = self.to_dir                
            logger.normal_logger.info(f'[LOADER] TO_DIR INPUT REQUIRED - Default Path:{to_dir}')

        r"""---------- Initialzing Timezone ----------""" 
        format_time_full = '%Y-%m-%d %H:%M:%S.%f'
        format_time_date = '%Y-%m-%d'
        if country == 'united states':
            today = datetime.datetime.now(timezone('US/Eastern'))
            tz = timezone('US/Eastern')
            now = datetime.datetime.now(tz)
            now_open = tz.localize(datetime.datetime(now.year, now.month, now.day, 9, 30, 0, 0))
            now_close = tz.localize(datetime.datetime(now.year, now.month, now.day, 16, 0, 0, 0))
        if country == 'korea':
            today = datetime.datetime.now(timezone('Asia/Seoul'))
            tz = timezone('Asia/Seoul')
            now = datetime.datetime.now(tz)
            now_open = tz.localize(datetime.datetime(now.year, now.month, now.day, 9, 0, 0, 0))
            now_close = tz.localize(datetime.datetime(now.year, now.month, now.day, 16, 0, 0, 0))
 
        r"""--------- Initializing UPDATE log directoreis ----------"""
        if update_log_dir:
            if not os.path.isdir(update_log_dir):
                os.mkdir(update_log_dir)

        if not update_log_dir:
            update_log_dir = self.update_log_dir
            logger.normal_logger.info(f'[LOADER] UPDATE_LOG_DIR INPUT REQUIRED - Default Path:{update_log_dir}')
            if not os.path.isdir(update_log_dir):
                os.mkdir(update_log_dir)
        
        r"""---------- Initializing UPDATE log file name -----------"""
        if not update_log_file:
            update_log_key = f'ohlcv_{interval}'
            update_log_file = self.update_log_dict[update_log_key]
            logger.normal_logger.info(f'[LOADER] UPDATE_LOG_FILE INPUT REQUIRED - Default Path:{update_log_file}')

        r"""---------- Initializing UPDATE log file ----------"""
        
        if not os.path.isfile(os.path.join(update_log_dir, update_log_file)):
            logger.normal_logger.info(f'[LOADER] UPDATE_LOG_FILE DOES NOT EXIST - Make {update_log_file} in {update_log_dir}')
            with open(os.path.join(update_log_dir, update_log_file), 'w') as log:
                json.dump(json.dumps(dict(), indent=4), log)
            update_log = dict()            
            for existed_security in map(lambda x: x[:-4], filter(lambda x: x[-3:] == 'csv', os.listdir(from_dir))):
                update_log[existed_security] = {'WhenDownload':today.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                                'WhenDownload_TZ':today.tzname(),
                                                'HowDownload':'origin',
                                                'Table_NumRows':None,
                                                'Table_NumColumns':None,
                                                'Table_Start':None,
                                                'Table_End':None,
                                                }         
            with open(os.path.join(update_log_dir, update_log_file), 'w') as log:
                json.dump(json.dumps(update_log, indent=4), log)
        with open(os.path.join(update_log_dir, update_log_file), 'r') as log:
            update_log = json.dump(json.loads(json.load(log)
        r"""---------- Double check for UPDATE log file ---------"""
        if not update_log:
            logger.normal_logger.info('[LOADER] UPDATE_LOG_FILE IS EMPTY - Rewrite with exisiting directories')
            update_log = dict()            
            for existed_security in map(lambda x: x[:-4], filter(lambda x: x[-3:] == 'csv', os.listdir(from_dir))):
                update_log[existed_security] = {'WhenDownload':today.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                                'WhenDownload_TZ':today.tzname(),
                                                'HowDownload':'origin',
                                                'Table_NumRows':None,
                                                'Table_NumColumns':None,
                                                'Table_Start':None,
                                                'Table_End':None,
                                                }         
            with open(os.path.join(update_log_dir, update_log_file), 'w') as log:
                json.dump(json.dumps(update_log, indent=4), log)    

        r"---------- Initializing SELECT baskets ----------"
        ## Update log file loading    
        with open(os.path.join(update_log_dir, update_log_file), 'r') as log:
            update_log = json.loads(json.load(log))

        ### Case 1) -> No Baskets    
        if not baskets:
            serialized_objects = os.listdir(from_dir)
            serialized_object =list(filter(lambda x: x[-3:] == 'csv' and '_' not in x, serialized_objects))
            baskets_in_dir = list(map(lambda x: x[:-4], serialized_object))
            if not baskets_in_dir:
                logger.normal_logger.info["[LOADER] NO BASKETS INPUT & NO BASKETS IN DIR"]
                return
            baskets_info = list(map(update_log.get, baskets_in_dir))
            baskets_dates = [value["Table_End"] for value in baskets_info]
            select_baskets = baskets_in_dir
            logger.normal_logger.info(f'[LOADER] NO BASKETS INPUT -> Baskets {baskets_in_dir} from {from_dir}')

            format_time_full = format_time_full ; now = now ; now_open = now_open ; now_close = now_close
            
            if None in baskets_dates:
                logger.normal_logger.info(f'[LOADER] ONE OF TICKERS IN THE BASETS HAS NO TIME RECORDS - Update All:{select_baskets}.')
            if not None in baskets_dates:
                try:
                    max_time = max(list(map(lambda x: tz.localize(datetime.datetime.strptime(x, format_time_full)), baskets_dates)))
                except ValueError: 
                    max_time = max(list(map(lambda x: tz.localize(datetime.datetime.strptime(x, format_time_date)), baskets_dates)))
                if interval == '1d':
                    max_time_close = tz.localize(datetime.datetime(max_time.year, max_time.month, max_time.day, now_close.hour, now_close.minute, now_close.second, now_close.microsecond))
                    logger.normal_logger.info('[LOADER] INTERVAL BASED ON 1D')
                    if ((((now - max_time_close).days == 1)) and (now > now_close)) or ((now - max_time_close).days >=2):
                        logger.normal_logger.info(f'[LOADER] BASKETS NEEDS UPDATE')    
                    else: 
                        logger.normal_logger.info(f'[LOADER] BASKETS ALL UP-TO-DATE - Loading {select_baskets} from Local {from_dir}')
                        datavendor = DataVendor(baskets=select_baskets, country=country)
                        return datavendor.ohlcv_from_local(baskets=select_baskets, from_dir=from_dir, update_log_dir=update_log_dir, update_log_file=update_log_file)
                if interval != '1d':
                    logger.normal_logger.info(f'[LOADER] INTERVAL BASED ON <> 1D -> TBD')
                    if (now - max_time):
                        logger.normal_logger.info(f'[LOADER] BASKETS NEEDS UPDATE')
                    else:
                        logger.normal_logger.info(f'[LOADER] BASKETS ALL UP-TO-DATE - Loading {select_baskets} from Local {from_dir}')
                        datavendor = DataVendor(baskets=select_baskets, country=country)
                        return datavendor.ohlcv_from_local(baskets=select_baskets, from_dir=from_dir, update_log_dir=update_log_dir, update_log_file=update_log_file)
     
        ### Case 2) -> Baskets
          
        ### Case 2-1) One of basekts are not in the log before or Log is empty -> SELECT baskets are all the tickers in the bakset
        if baskets:
            baskets_in_log = update_log.keys()
            new_baskets = list(filter(lambda x: x not in baskets_in_log, baskets))
            old_baskets = list(filter(lambda x: x in baskets_in_log, baskets))
            old_baskets_info = list(map(update_log.get, old_baskets))
            old_baskets_dates = [value["Table_End"] for value in old_baskets_info]
            old_baskets_local = dict()
            if None in old_baskets_dates:
                select_baskets = new_baskets + old_baskets
                logger.normal_logger.info(f'[LOADER] ONE OF TICKERS IN THE BASETS HAS NO TIME RECORDS - Update All:{select_baskets}.')    
            if not None in old_baskets_dates:
                format_time_full = format_time_full ; now = now ; now_open = now_open ; now_close = now_close
                try:
                    max_time = max(list(map(lambda x: tz.localize(datetime.datetime.strptime(x, format_time_full)), old_baskets_dates)))
                except ValueError:
                    max_time = max(list(map(lambda x: tz.localize(datetime.datetime.strptime(x, format_time_date)), old_baskets_dates)))
                if interval == '1d':
                        max_time_close = tz.localize(datetime.datetime(max_time.year, max_time.month, max_time.day, now_close.hour, now_close.minute, now_close.second, now_close.microsecond))
                        logger.normal_logger.info('[LOADER] INTERVAL BASED ON 1D')
                        if ((((now - max_time_close).days == 1)) and (now > now_close)) or ((now - max_time_close).days >=2):
                            if not new_baskets:
                                select_baskets = old_baskets        
                                logger.normal_logger.info(f'[LOADER] NO NEW BASKETS & OLD BASKETS NEEDS UPDATE')
                            if new_baskets:
                                select_baskets = new_baskets + old_baskets
                                logger.normal_logger.info(f'[LOADER] NEW BASKETS & OLD BASKETS NEEDS UPDATE')        
                        else: 
                            if not new_baskets:
                                select_baskets = old_baskets
                                logger.normal_logger.info(f'[LOADER] NO NEW BASKETS & OLD BASKETS UP-TO-DATE')
                                datavendor_for_local = DataVendor(baskets=select_baskets, country=country)
                                return datavendor_for_local.ohlcv_from_local(baskets=select_baskets, from_dir=from_dir, update_log_dir=update_log_dir, update_log_file=update_log_file)        
                            if new_baskets:
                                logger.normal_logger.info(f'[LOADER] NEW BASKETS NEEDS UPDATE & OLD BASKETS UP-TO-DATE - Loading {select_baskets} from Local {from_dir}')
                                datavendor_for_local = DataVendor(baskets=select_baskets, country=country)
                                old_baskets_local = datavendor_for_local.ohlcv_from_local(baskets=select_baskets, from_dir=from_dir, update_log_dir=update_log_dir, update_log_file=update_log_file)        
                                select_baskets = new_baskets
                
                if interval != '1d':
                    logger.normal_logger.info(f'[LOADER] INTERVAL BASED ON != 1D -> TBD')
                    if (now - max_time):
                        logger.normal_logger.info(f'[LOADER] BASKETS NEEDS UPDATE')
                        if not new_baskets:
                                select_baskets = old_baskets        
                                logger.normal_logger.info(f'[LOADER] NO NEW BASKETS & OLD BASKETS NEEDS UPDATE')
                        if new_baskets:
                            select_baskets = new_baskets + old_baskets
                            logger.normal_logger.info(f'[LOADER] NEW BASKETS & OLD BASKETS NEEDS UPDATE')      
                    else: 
                        if not new_baskets:
                            select_baskets = old_baskets
                            logger.normal_logger.info(f'[LOADER] NO NEW BASKETS & OLD BASKETS UP-TO-DATE')
                            datavendor_for_local = DataVendor(baskets=select_baskets, country=country)
                            return datavendor_for_local.ohlcv_from_local(baskets=select_baskets, from_dir=from_dir, update_log_dir=update_log_dir, update_log_file=update_log_file)        
                        if new_baskets:
                            logger.normal_logger.info(f'[LOADER] NEW BASKETS NEEDS UPDATE & OLD BASKETS UP-TO-DATE - Loading {select_baskets} from Local {from_dir}')
                            datavendor_for_local = DataVendor(baskets=select_baskets, country=country)
                            old_baskets_local = datavendor_for_local.ohlcv_from_local(baskets=select_baskets, from_dir=from_dir, update_log_dir=update_log_dir, update_log_file=update_log_file)        
                            select_baskets = new_baskets

     
        r""" ---------- Executing DataVendor ----------"""   
        logger.normal_logger.info("[LOADER] EXECUTING DATAVENDOR:{select_baskets}".format(select_baskets=select_baskets))
        datavendor = DataVendor(baskets=select_baskets, country=country)
        
        r""" --------- ohlcv from yahooquery ----------"""
        if interval == '1d':
            logger.normal_logger.info('[LOADER] TBD * from yahooquery currently only supporting \'1d\' frequency. Please make sure dataset directories are seperate by frequency')

        if source == 'yahooquery':
            logger.normal_logger.info('[LOADER] * from yahooquery')
            if old_baskets_local:
                new_baskets = datavendor.ohlcv_from_yahooquery(baskets=select_baskets, from_dir=from_dir, to_dir=to_dir, 
                                                    update_log_dir=update_log_dir, update_log_file=update_log_file, interval=interval, country=country, progress=True)
                new_baskets.dict.update(old_baskets_local.dict)
                return new_baskets
            if  not old_baskets_local:
                return datavendor.ohlcv_from_yahooquery(baskets=select_baskets, from_dir=from_dir, to_dir=to_dir, 
                                                    update_log_dir=update_log_dir, update_log_file=update_log_file, interval=interval, country=country, progress=True)
            
        r""" ---------- ohlcv from fdr reader ----------"""
        if source == 'fdr':
            logger.normal_logger.info('[LOADER] * from finance-datareader')
            if old_baskets_local:
                new_baskets = datavendor.ohlcv_from_fdr(baskets=select_baskets, from_dir=from_dir, to_dir=to_dir, 
                                                    update_log_dir=update_log_dir, update_log_file=update_log_file, interval=interval, country=country, progress=True)
                new_baskets.dict.update(old_baskets_local.dict)
                return new_baskets
            if  not old_baskets_local:
                return datavendor.ohlcv_from_fdr(baskets=select_baskets, from_dir=from_dir, to_dir=to_dir, 
                                                    update_log_dir=update_log_dir, update_log_file=update_log_file, interval=interval, country=country, progress=True)
            


    def fundamentals_loader(self, baskets:Iterable[str], from_dir=dataset_dirname, to_dir=dataset_dirname, 
                    update_log_dir=None, update_log_file=None, country='united states', modules=None, frequency = None, source='yahooquery'):
        r"""---------- Initializing modules ----------"""
        if not modules:
            modules = list(self.fmf)
            logger.normal_logger.info("[LOADER] MODULES INPUT[LIST, STR] REQUIRED - Default Modules: {modules}")
        r"""---------- Initialzing dataset directories ----------"""
        if not from_dir:
            from_dir = self.from_dir    
            logger.normal_logger.info(f'[LOADER] FROM_DIR INPUT REQUIRED - Default Path:{from_dir}')
        if not to_dir:
            to_dir = self.to_dir           
            logger.normal_logger.info(f'[LOADER] TO_DIR INPUT REQUIRED - Default Path:{to_dir}')                      
        r"""---------- Initialzing Timezone ----------"""
        format_time_full = '%Y-%m-%d %H:%M:%S.%f'
        format_time_date = '%Y-%m-%d'
        if country == 'united states':
            today = datetime.datetime.now(timezone('US/Eastern'))
            tz = timezone("US/Eastern")
            now = datetime.datetime.now(tz)
        if country == 'korea':
            today = datetime.datetime.now(timezone('Asia/Seoul'))
            tz = timezone("Asia/Seoul")
            now = datetime.datetime.now(tz)
        r"""--------- Initializing UPDATE log directoreis ----------"""
        if update_log_dir:
            if not os.path.isdir(update_log_dir):
                os.mkdir(update_log_dir)
        if not update_log_dir:
            update_log_dir = self.update_log_dir
            logger.normal_logger.info(f'[LOADER] UPDATE_LOG_DIR INPUT REQUIRED - Default Path:{update_log_dir}')
        if not os.path.isdir(update_log_dir):
            os.mkdir(update_log_dir)
        r"""---------- Initializing UPDATE log file name -----------"""
        if not update_log_file:
            update_log_key = f'fundamentals'
            update_log_file = self.update_log_dict[update_log_key]
            logger.normal_logger.info(f'[LOADER] UPDATE_LOG_FILE INPUT REQUIRED - Default Path:{update_log_file}')
        r"""---------- Initializing UPDATE log file ----------"""
        if not os.path.isfile(os.path.join(update_log_dir, update_log_file)):
            logger.normal_logger.info(f'[LOADER] UPDATE_LOG_FILE DOES NOT EXIST - Make {update_log_file} in {update_log_dir}')
            with open(os.path.join(update_log_dir, update_log_file), 'w') as log:
                json.dump(json.dumps(dict(), indent=4), log)
                    
            update_log = dict()            
            baskets_in_dir = [existed_security for existed_security in map(lambda x: x[:-17], filter(lambda x: x[-16:] == 'fundamentals_csv', os.listdir(from_dir)))]
            update_log['Modules'] = list()
            update_log['WhenDownload'] = today.strftime('%Y-%m-%d %H:%M:%S.%f')
            update_log['WhenDownload_TZ'] = today.tzname()
            update_log['Baskets'] = baskets_in_dir
                     
            with open(os.path.join(update_log_dir, update_log_file), 'w') as log:
                json.dump(json.dumps(update_log, indent=4), log)
        with open(os.path.join(update_log_dir, update_log_file), 'r') as log:
            update_log = json.loads(json.load(log) 
        r"""---------- Double check for UPDATE log file ---------"""
        if not update_log:
            logger.normal_logger.info('[LOADER] UPDATE_LOG_FILE IS EMPTY - Rewrite with exisiting directories')
            with open(os.path.join(update_log_dir, update_log_file), 'w') as log:
                json.dump(json.dumps(dict(), indent=4), log)
                    
            update_log = dict()            
            baskets_in_dir = [existed_security for existed_security in map(lambda x: x[:-17], filter(lambda x: x[-16:] == 'fundamentals_csv', os.listdir(from_dir)))]
            update_log['Modules'] = list()
            update_log['WhenDownload'] = today.strftime('%Y-%m-%d %H:%M:%S.%f')
            update_log['WhenDownload_TZ'] = today.tzname()
            update_log['Baskets'] = baskets_in_dir
                     
            with open(os.path.join(update_log_dir, update_log_file), 'w') as log:
                json.dump(json.dumps(update_log, indent=4), log)
        r"---------- Initizlizing frequency ----------"
        if not frequency:
            frequency = 2
            logger.normal_logger.info(f"[LOADER] DEFAULT FREQUENCY {frequency}")

        r"---------- Initializing SELECT baskets ----------"
        ## Update log file loading    
        with open(os.path.join(update_log_dir, update_log_file), 'r') as log:
            update_log = json.loads(json.load(log))

        ### Case 1) -> No Baskets    
        if not baskets:
            try:
                baskets_in_csv = pd.read_csv(os.path.join(from_dir, 'fundamentals.csv'))['ticker']
            except FileNotFoundError:
                logger.normal_logger.info["[LOADER] NO BASKETS INPUT & NO FUNDAMENTALS BASKETS CSV"]
                return
            baskets_in_log = update_log['Baskets']
            new_baskets = list(filter(lambda x: x not in baskets_in_log, baskets_in_csv))
            old_baskets = list(filter(lambda x: x in baskets_in_log, baskets_in_csv))
            if modules != update_log['Modules']:
                select_baskets = new_baskets+old_baskets
                logger.normal_logger.info(f'[LOADER] MODULES {modules} CHANGED ALL REUPDATE')
            if modules == update_log['Modules']:
                format_time_full = format_time_full ; now = now ; log_date = tz.localize(datetime.datetime.strptime(update_log['WhenDownload'], format_time_full))
                if (now - log_date) >= frequency:
                    select_baskets = new_baskets + old_baskets
                    logger.normal_logger.info(f'[LOADER] BASKETS OUTDATED BY {frequency} ALL REUPDATE')
                if (now - log_date) < frequency:
                    select_baskets = new_baskets
                    if not select_baskets:
                        logger.normal_logger.info(f'[LOADER] ALL BASKETS UP-TO-DATE. LOAD FROM LOCAL {from_dir}')
                        datavendor = DataVendor(baskets=select_baskets, country=country)
                        return datavendor.fundamentals_from_local(baskets=select_baskets, from_dir=from_dir, update_log_dir=update_log_dir, update_log_file=update_log_file)
                    logger.normal_logger.info(f'[LOADER] BASKETS UP-TO-DATE NEW BASKETS {select_baskets} UPDATE')
        ### Case 2) -> Baskets
        if baskets:
            baskets_in_log = update_log['Baskets']
            new_baskets = list(filter(lambda x: x not in baskets_in_log, baskets))
            old_baskets = list(filter(lambda x: x in baskets_in_log, baskets))
            if modules != update_log['Modules']:
                select_baskets = new_baskets+old_baskets
                logger.normal_logger.info(f'[LOADER] MODULES {modules} CHANGED ALL REUPDATE')
            if modules == update_log['Modules']:
                format_time_full = format_time_full ; now = now ; log_date = tz.localize(datetime.datetime.strptime(update_log['WhenDownload'], format_time_full))
                if (now - log_date) >= frequency:
                    select_baskets = new_baskets + old_baskets
                    logger.normal_logger.info(f'[LOADER] BASKETS OUTDATED BY {frequency} ALL REUPDATE')
                if (now - log_date) < frequency:
                    select_baskets = new_baskets
                    if not select_baskets:
                        logger.normal_logger.info(f'[LOADER] ALL BASKETS UP-TO-DATE. LOAD FROM LOCAL {from_dir}')
                        datavendor = DataVendor(baskets=select_baskets, country=country)
                        return datavendor.fundamentals_from_local(baskets=select_baskets, from_dir=from_dir, update_log_dir=update_log_dir, update_log_file=update_log_file)
                    logger.normal_logger.info(f'[LOADER] BASKETS UP-TO-DATE NEW BASKETS {select_baskets} UPDATE')             

        r""" ---------- Executing DataVendor ----------"""   
        logger.normal_logger.info("[LOADER] EXECUTING DATAVENDOR :{select_baskets}".format(select_baskets=select_baskets))
        datavendor = DataVendor(baskets=select_baskets, country=country)
        
        r""" --------- fundamentals from yahooquery ----------"""
        if source == 'yahooquery':
            logger.normal_logger.info('[LOADER] * from yahooquery')
            return datavendor.fundamentals_from_yahooquery(baskets=select_baskets, from_dir=from_dir, to_dir=to_dir, 
                                            update_log_dir=update_log_dir, update_log_file=update_log_file, country=country, modules=modules)
            
                        
        






