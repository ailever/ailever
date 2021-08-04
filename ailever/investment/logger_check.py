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

    if not os.path.isfile(os.path.join(log_path, log_file)):
        return integrated_dataloader(baskets, path, source)
        

    with open(os.path.join(log_path, log_file),'r') as log:
            download_log = json.loads(json.load(log))
    
    if not baskets in download_log.keys():
        return integrated_dataloader(baskets, path, source)
        

    in_basket_values = list(map(download_log.get, baskets))
    in_basket_dates = [value["Table_EndDate"] for value in in_basket_values]

    if None in in_basket_dates:
        return integrated_dataloader(baskets, path, source)
        
    format_time = '%Y-%m-%d'
    if datetime.now(timezone('US/Eastern')) > max(list(map(lambda x: datetime.strptime(x, format_time), in_basket_dates))):
        return integrated_dataloader(baskets, path, source)
        

