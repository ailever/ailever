from ..path import refine

import os
import json
import pandas as pd
import FinanceDataReader as fdr

def integrated_loader(baskets, path=False):
    if path:
        if loader.firstcall:
            loader.firstcall = False
            loader._initialize(dataset_dirname=refine(path))
            
    with open('.dataset_log.json', 'r') as log:
        download_log = json.loads(json.load(log))
    
    existed_securities = filter(lambda x: x in download_log, baskets)
    not_existed_securities = filter(lambda x: not x in download_log, baskets)

    loader.from_fdr(not_existed_securities)
    if not bool(loader.failures):
        return loader.from_local(baskets)
    else:
        print('[AILEVER] Download failure list: ', loader.failures)
        return loader.from_local(loader.successes)

class Loader:
    def __init__(self):
        self.firstcall = True
        self.dataset_dirname = '.financedatasets'
        self.log_filename = '.dataset_log.json'
        self.successes = set()
        self.failures = set()
        self._initialize()
    
    def _initialize(self, dataset_dirname=False):
        if dataset_dirname:
            self.dataset_dirname = dataset_dirname

        if not os.path.isdir(self.dataset_dirname):
            os.mkdir(self.dataset_dirname)

        with open(self.log_filename, 'w') as log:
            json.dump(json.dumps(dict(), indent=4), log)
        with open(self.log_filename, 'r') as log:
            download_log = json.loads(json.load(log))

        for existed_security in map(lambda x: x[:-4], filter(lambda x: x[-3:] == 'csv', os.listdir(self.dataset_dirname))):
            download_log[existed_security] = 'origin'

        with open(self.log_filename, 'w') as log:
            json.dump(json.dumps(download_log, indent=4), log)

        self.successes.update(download_log.keys())

    def from_local(self, baskets):
        dataset = dict()
        for security in baskets: 
            dataset[security] = pd.read_csv(os.path.join(self.dataset_dirname, f'{security}.csv'))
        return dataset

    def from_fdr(self, baskets):
        successes = list()
        failures = list()
        for security in baskets:
            try:
                fdr.DataReader(security).to_csv(os.path.join(self.dataset_dirname, f'{security}.csv'))
                successes.append(security)
            except:
                failures.append(security)
                continue

        self.successes.update(successes)
        self.failures.update(failures)
        self._logger_for_successes('from_fdr')
        
    def _logger_for_successes(self, message):
        with open(self.log_filename, 'r') as log:
            download_log = json.loads(json.load(log))

        for successed_security in self.successes:
            download_log[successed_security] = message
        
        with open(self.log_filename, 'w') as log:
            json.dump(json.dumps(download_log, indent=4), log)

loader = Loader()
