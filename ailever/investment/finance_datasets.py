import os
import json
import pandas as pd
import FinanceDataReader as fdr

def integrated_loader(baskets):
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
        self.dataset_dirname = '.financedatasets'
        self.log_filename = '.dataset_log.json'
        self.successes = set()
        self.failures = set()

        if not os.path.isdir(self.dataset_dirname):
            os.path.mkdir(self.dataset_dirname)
            with open(self.log_filename, 'w') as log:
                json.dump(json.dumps(dict(), indent=4), log)
        else:
            if not os.path.isfile(self.log_filename):
                with open(self.log_filename, 'w') as log:
                    json.dump(json.dumps(dict(), indent=4), log)
            else:
                pass
        
        with open(self.log_filename, 'r') as log:
            download_log = json.loads(json.load(log))
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
