from ailever.investment import fmlops_bs

from ...path import refine
from .._base_transfer import DataTransferCore
from ..logger import update_log
from ..logger import Logger
from .DataVendor import DataVendor

from typing import Optional, Any, Union, Callable, Iterable
from pytz import timezone

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
base_dir['rawdata_repository'] = fmlops_bs.local_system.root.rawdata_repository.name
base_dir['metadata_store'] = fmlops_bs.local_system.root.metadata_store.name
base_dir['feature_store'] = fmlops_bs.local_system.root.feature_store.name
base_dir['model_registry'] = fmlops_bs.local_system.root.model_registry.name
base_dir['source_repotitory'] = fmlops_bs.local_system.root.source_repository.name

logger = Logger()
dataset_dirname = os.path.join(base_dir['root'], base_dir['rawdata_repository'])
log_dirname = os.path.join(base_dir['root'], base_dir['metadata_store'])
update_log_file = update_log


class Loader():

    baskets = None
    from_dir = None
    to_dir = None
    fundamentals_modules_fromyahooquery = DataVendor.fundamentals_modules_fromyahooquery
    fmf = DataVendor.fundamentals_modules_fromyahooquery

    def __init__(self, baskets=None):

        self.baskets = baskets
        self.from_dir = dataset_dirname 
        self.to_dir =  dataset_dirname
        self.update_log_dir = log_dirname
        self.update_log_file = update_log_file['ohlcv']


    def ohlcv_loader(self, baskets:Iterable[str]=None, from_dir=None, to_dir=None, 
                    update_log_dir=None, update_log_file=None, country='united states', interval='1d', source='yahooquery'):

        r"""---------- Initialzing dataset directories ----------"""

        if not from_dir:
            from_dir = self.from_dir    
            logger.normal_logger.info(f'FROM_DIR INPUT REQUIRED - Default Path:{from_dir}')
        if not to_dir:
            to_dir = self.to_dir                
            logger.normal_logger.info(f'TO_DIR INPUT REQUIRED - Default Path:{to_dir}')

        r"""---------- Initialzing Timezone ----------""" 

        if country == 'united states':
            today = datetime.datetime.now(timezone('US/Eastern'))
            tz = timezone('US/Eastern')
        if country == 'korea':
            today = datetime.datetime.now(timezone('Asia/Seoul'))
            tz = timezone('Asia/Seoul')

        r"""--------- Initializing UPDATE log directoreis ----------"""

        if update_log_dir:
            if not os.path.isdir(update_log_dir):
                os.mkdir(update_log_dir)

        if not update_log_dir:
            update_log_dir = self.update_log_dir
            logger.normal_logger.info(f'UPDATE_LOG_DIR INPUT REQUIRED - Default Path:{update_log_dir}')
            if not os.path.isdir(update_log_dir):
                os.mkdir(update_log_dir)
        
        r"""---------- Initializing UPDATE log file ----------"""
        
        if not update_log_file:
            update_log_file = self.update_log_file
            logger.normal_logger.info(f'UPDATE_LOG_FILE INPUT REQUIRED - Default Path:{update_log_file}')
        
        if not os.path.isfile(os.path.join(update_log_dir, update_log_file)):
            logger.normal_logger.info(f'UPDATE_LOG_FILE DOES NOT EXIST - Make {update_log_file} in {update_log_dir}')
            with open(os.path.join(update_log_dir, update_log_file), 'w') as log:
                json.dump(json.dumps(dict(), indent=4), log)
            update_log = dict()            
            for existed_security in map(lambda x: x[:-4], filter(lambda x: x[-3:] == 'csv', os.listdir(from_dir))):
                update_log[existed_security] = {'WhenDownload':today.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                                'WhenDownload_TZ':today.tzname(),
                                                'HowDownload':'origin',
                                                'Table_NumRows':None,
                                                'Table_NumColumns':None,
                                                'Table_StartDate':None,
                                                'Table_EndDate':None,
                                                }         
            with open(os.path.join(update_log_dir, update_log_file), 'w') as log:
                json.dump(json.dumps(update_log, indent=4), log)
         
        with open(os.path.join(update_log_dir, update_log_file), 'r') as log:
            update_log = json.loads(json.load(log))
        
        if not update_log:
            logger.normal_logger.info('UPDATE_LOG_FILE IS EMPTY - Rewrite with exisiting directories')
            update_log = dict()            
            for existed_security in map(lambda x: x[:-4], filter(lambda x: x[-3:] == 'csv', os.listdir(from_dir))):
                update_log[existed_security] = {'WhenDownload':today.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                                'WhenDownload_TZ':today.tzname(),
                                                'HowDownload':'origin',
                                                'Table_NumRows':None,
                                                'Table_NumColumns':None,
                                                'Table_StartDate':None,
                                                'Table_EndDate':None,
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
            serialized_object =list(filter(lambda x: x[-3:] == 'csv', serialized_objects))
            baskets_in_dir = list(map(lambda x: x[:-4], serialized_objects))
            
            tickers_in_dir = list(map(update_log.get, baskets_in_dir))
            tickers_dates = [value["WhenDownload"] for value in tickers_in_dir]
            
            format_time = '%Y-%m-%d %H:%M:%S.%f'
            if datetime.datetime.now(tz) > max(list(map(lambda x: tz.localize(datetime.datetime.strptime(x, format_time)), tickers_dates))):
                select_baskets = tickers_in_dir
                logger.normal_logger.info(f'BASKETS INPUT REQUIRED - Default Basket:{select_baskets} in the directory:{from_dir}.')    
            else: 
                logger.normal_logger.info(f'BASKETS INPUT REQUIRED - Default Basket:{baskets} in the directory:{from_dir} Are All UP-TO-DATE')    
                return

        ### Case 2) -> Baskets
          
        ### Case 2-1) One of basekts are not in the log before or Log is empty -> SELECT baskets are all the tickers in the bakset
        if ((set(baskets) & set(list(update_log.keys()))) != set(baskets)) or (not update_log):
            select_baskets =  baskets
            logger.normal_logger.info(f'ONE OF TICKERS IN THE BASKETS ARE NEW OR LOG IS EMPTY - Update All:{select_baskets}.')      
        else:
            in_the_baskets = list(map(update_log.get, baskets))
            tickers_dates = [value["WhenDownload"] for value in in_the_baskets]          
        ### Case 2-2) One of Tickers has no time records
            if None in tickers_dates:
                select_baskets = baskets
                logger.normal_logger.info(f'ONE OF TICKERS IN THE BASETS HAS NO TIME RECORDS - Update All:{select_baskets}.')    
        ### Case 2-3) all tickers in basket was in exisitng logger but they are outdated
            if (set(baskets) & set(list(update_log.keys()))) == set(baskets):
                in_the_baskets = list(map(update_log.get, baskets))
                tickers_dates = [value["WhenDownload"] for value in in_the_baskets]
                format_time = '%Y-%m-%d %H:%M:%S.%f'
                if datetime.datetime.now(tz) > max(list(map(lambda x: tz.localize(datetime.datetime.strptime(x, format_time)), tickers_dates))):
                    select_baskets = baskets     
                    logger.normal_logger.info(f'BASETS OUTDATED - Update All:{select_baskets}.')

        r""" ---------- Executing DataVendor ----------"""   
        logger.normal_logger.info("EXECUTING DATAVENDOR:{select_baskets}".format(select_baskets=select_baskets))
        datavendor = DataVendor(baskets=select_baskets, country=country)
        
        r""" --------- ohlcv from yahooquery ----------"""
        if interval == '1d':
            logger.normal_logger.info('TBD * from yahooquery currently only supporting \'1d\' frequency. Please make sure dataset directories are seperate by frequency')

        if source == 'yahooquery':
            logger.normal_logger.info('* from yahooquery')
            datavendor.ohlcv_from_yahooquery(baskets=select_baskets, from_dir=from_dir, to_dir=to_dir, 
                                                update_log_dir=update_log_dir, update_log_file=update_log_file, interval=interval, country=country, progress=True)
            
            if not bool(datavendor.failures):
                return datavendor.ohlcv_from_local(baskets=select_baskets, from_dir=from_dir, to_dir=to_dir, update_log_dir=update_log_dir, update_log_file=update_log_file)

            else:
                return datavendor.ohlcv_from_local(baskets=datavendor.successes, from_dir=from_dir, to_dir=to_dir, update_log_dir=update_log_dir, update_log_file=update_log_file)
            
        r""" ---------- ohlcv from fdr reader ----------"""
        logger.normal_logger.info('* from finance-datareader')
        datavendor.from_fdr(baskets=baskets, from_dir=from_dir, to_dir=to_dir, 
                                update_log_dir=update_log_dir, update_log_file=update_log_file, interval=interval, country=country)
        if not bool(datavendor.failures):
            return datavendor.ohlcv_from_local(baskets=baskets, from_dir=from_dir, to_dir=to_dir, update_log_dir=update_log_dir, update_log_file=update_log_file)
        else:
            return datavendor.ohlcv_from_local(baskets=datavendor.successes, from_dir=from_dir, to_dir=to_dir, update_log_dir=update_log_dir, update_log_file=update_log_file) 
               

    def fundamentals_loader(self, baskets:Iterable[str], from_dir=dataset_dirname, to_dir=dataset_dirname, 
                    update_log_dir=None, update_log_file=None, country='united states', modules=None ,source='yahooquery'):
        r"""---------- Initializing modules ----------"""
        if not modules:
            logger.normal_logger.info("MODULES INPUT[LIST, STR] REQUIRED - Default Modules: DividendYield")
            modules = 'DividendYield'

        r"""---------- Initialzing dataset directories ----------"""
        if not from_dir:
            from_dir = self.from_dir    
            logger.normal_logger.info(f'FROM_DIR INPUT REQUIRED - Default Path:{from_dir}')
        if not to_dir:
            to_dir = self.to_dir           
            logger.normal_logger.info(f'TO_DIR INPUT REQUIRED - Default Path:{to_dir}')                      
        r"""---------- Initialzing Timezone ----------"""
        if country == 'united states':
            today = datetime.datetime.now(timezone('US/Eastern'))
            tz = timzone("US/Eastern")
        if country == 'korea':
            today = datetime.datetime.now(timezone('Asia/Seoul'))
            tz = timezone("Asia/Seoul")
        r""" ---------- Executing DataVendor ----------"""   
        logger.normal_logger.info("EXECUTING DATAVENDOR :{baskets}".format(baskets=baskets))
        datavendor = DataVendor(baskets=baskets, country=country)
        
        r""" --------- fundamentals from yahooquery ----------"""
        if source == 'yahooquery':
            logger.normal_logger.info('* from yahooquery')
            return datavendor.fundamentals_from_yahooquery(baskets=baskets, from_dir=from_dir, to_dir=to_dir, 
                                            update_log_dir=update_log_dir, update_log_file=update_log_file, country=country, modules=modules)
            
                        
        






