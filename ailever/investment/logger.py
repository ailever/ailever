
from ailever.investment._fmlops_policy import fmlops_bs

import logging
import logging.config
import os

import FinanceDataReader as fdr

base_dir = dict()
base_dir['root'] = fmlops_bs.local_system.root
base_dir['rawdata_repository'] = fmlops_bs.local_system.rawdata_repository
base_dir['metadata_store'] = fmlops_bs.local_system.metadata_store
base_dir['feature_store'] = fmlops_bs.local_system.feature_store
base_dir['model_registry'] = fmlops_bs.local_system.model_registry
base_dir['source_repotitory'] = fmlops_bs.local_system.source_repository

log_dirname = os.path.join(base_dir['root'], base_dir['metadata_store'])
update_log = {'ohlcv':'ohlcv.json'}

class Logger():
    
    config = {
            "version": 1,
            "formatters": {
                "simple": {"format": "[%(name)s] %(message)s"},
                "complex":{
                    "format": "[%(asctime)s]/[%(name)s]/[%(filename)s:%(lineno)d]/[%(levelname)s]/[%(message)s]"},
                },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "level": "DEBUG",
                    },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(log_dirname, "meta.log"),
                    "formatter": "complex",
                    "level": "INFO",
                    },
                },
            "root": {"handlers": ["console", "file"], "level": "WARNING"},
            "loggers": {"normal": {"level": "INFO"}, "dev": {"level": "DEBUG"},},
            }


    def __init__(self):
        
        logging.config.dictConfig(self.config)       
        root_logger = logging.getLogger("root")
        normal_logger = logging.getLogger("normal")
        dev_logger = logging.getLogger("dev")

def ptest():

    for i in range(10):
        print(i)
        normal_logger.info(i)

def ptest2():

    lst = ['ARE','BXP','SPG', 'SLG']
    for ticker in lst:
        normal_logger.info(fdr.DataReader(ticker))

if __name__=="__main__":
    
    logger = Logger()

    ptest()
    ptest2()
        

