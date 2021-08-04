import json
import os
from datetime import datetime
from .finance_datasets import integrated_dataloader
from pytz import timezone

"""
EST 09:30 ~ 16:00
"""

def financedatasets_logger(baskets=False, path=False, source="yahooquery", log_file=".dataset_log.json", log_path=r"./"):
    
    assert baskets, "No baskets input will lead to no change"

    # Case 1) No log file - Initiate
    if not os.path.isfile(os.path.join(log_path, log_file)):
        return integrated_dataloader(baskets, path, source)
    
    with open(os.path.join(log_path, log_file),'r') as log:
            download_log = json.loads(json.load(log))
    
    # Case 2) When any tickers in baskets are not in existing logger -> tickers in baskets all renew
    if not baskets in list(download_log.keys()):
        return integrated_dataloader(baskets, path, source)
        
    in_basket_values = list(map(download_log.get, baskets))
    in_basket_dates = [value["Table_EndDate"] for value in in_basket_values]

    # Case 3) When no Table end date is put it (eg. when log file was newly made with in-place outsourced csv files)
    if None in in_basket_dates:
        return integrated_dataloader(baskets, path, source)
    
    # Case 4) all tickers in basket was in exisitng logger but they are outdated
    format_time = '%Y-%m-%d'
    if datetime.now(timezone('US/Eastern')) > max(list(map(lambda x: datetime.strptime(x, format_time), in_basket_dates))):
        return integrated_dataloader(baskets, path, source)
        

