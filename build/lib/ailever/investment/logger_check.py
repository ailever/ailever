from .finance_datasets import ohlcv_dataloader
from ._fmlops_policy import fmlops_bs

from datetime import datetime
from pytz import timezone
import json
import os

"""
EST 09:30 ~ 16:00
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

def ohlcv_update(baskets=None, path=dataset_dirname, source="yahooquery", log_file=".dataset_log.json", log_path=log_dirname):
        
    # Case 1) No log file - Initiate
    if not os.path.isfile(os.path.join(log_path, log_file)):
        return ohlcv_dataloader(baskets, path, source)
    
    with open(os.path.join(log_path, log_file),'r') as log:
            download_log = json.loads(json.load(log))
    
    # Case 2) When any tickers in baskets are not in existing logger -> tickers in baskets all renew
    if not baskets in list(download_log.keys()):
        return ohlcv_dataloader(baskets, path, source)
        
    in_basket_values = list(map(download_log.get, baskets))
    in_basket_dates = [value["Table_EndDate"] for value in in_basket_values]

    # Case 3) When no Table end date is put it (eg. when log file was newly made with in-place outsourced csv files)
    if None in in_basket_dates:
        return ohlcv_dataloader(baskets, path, source)
    
    # Case 4) all tickers in basket was in exisitng logger but they are outdated
    format_time = '%Y-%m-%d'
    if datetime.now(timezone('US/Eastern')) > max(list(map(lambda x: datetime.strptime(x, format_time), in_basket_dates))):
        return ohlcv_dataloader(baskets, path, source)
        

