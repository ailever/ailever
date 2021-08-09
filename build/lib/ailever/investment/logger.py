
from ._fmlops_policy import fmlops_bs

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

r"""---------- DEFAULT CONFIG for Logger and UPDATE log ----------"""

update_log = {'ohlcv':'ohlcv.json'}
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

r"""---------- SEE ABOVE FOR DEFAULT CONFIG ----------"""

class Logger():
     
    def __init__(self, config=config):
        
        self.config = config
        logging.config.dictConfig(self.config)       
        
        self.root_logger = logging.getLogger("root")
        self.normal_logger = logging.getLogger("normal")
        self.dev_logger = logging.getLogger("dev") 















def ptest():

    for i in range(10):
        logger.dev_logger.info(i)

def ptest2():

    lst = ['ARE','BXP','SPG', 'SLG']
    for ticker in lst:
        logger.normal_logger.info(ticker)

if __name__=="__main__":
    
    logger = Logger()

    ptest()
    ptest2()
        

