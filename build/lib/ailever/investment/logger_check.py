import json
import os
from datetime import datetime
from ailever.investment import integrated_loader


def status_finance_datasets(log_file=".dataset_log.json", log_path=r"./", baskets, path, on_assets):

    if not os.isfile(os.path.join(log_path, log_file):
        integrated_loader(baskets, path, on_assets)
    with open(os.path.join(log_path, log_file),'r') as log:
            download_log = json.loads(json.load(log))
    
    download_log_times = list(download_log.keys())
    format_time = '%Y-%m-%d %H:%M:%S'
    download_log_times = list(map(lambda x: datetime.strptime(x, format_time)

    if datetime.today().date() >= max(download_log_times).date():
        integrated_loader(baskets, path, on_assets)



