from ..path import refine

import os
import json
from tqdm import tqdm
import pandas as pd
import FinanceDataReader as fdr
from yahooquery import Ticker

def integrated_loader(baskets, path=False):
    if path:
        if path == '.financedatasets':
            pass
        elif loader.firstcall:
            loader.firstcall = False
            loader._initialize(dataset_dirname=refine(path))
    
        if loader.dataset_dirname != refine(path):
            loader._initialize(dataset_dirname=refine(path))

    with open('.dataset_log.json', 'r') as log:
        download_log = json.loads(json.load(log))
    
    existed_securities = filter(lambda x: x in download_log, baskets)
    not_existed_securities = filter(lambda x: not x in download_log, baskets)

    # priority 1 : yahooquery
    print('* from yahooquery')
    loader.from_yahooquery(baskets=not_existed_securities, country='united states', progress=True)
    print('[2]', loader.failures)
    if bool(loader.failures):
        # priority 2 : finance datareader
        print('* from finance-datareader')
        loader.from_fdr(baskets=loader.failures)
        if bool(loader.failures):
            # priority 3 : ?
            pass

    # Final Load : From Local
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

    def from_yahooquery(self, baskets, asynchronouse=False, backoff_factor=0.3, country='united states',
                        formatted=False, max_workers=8, proxies=None, retry=5, status_forcelist=[404, 429, 500, 502, 503, 504], timeout=5,
                        validate=False, verify=True, progress=True):
        successes = list()
        failures = list()
        try:
            ticker = Ticker(symbols=list(baskets), asynchronouse=asynchronouse, backoff_factor=backoff_factor, country=country,
                            formatted=formatted, max_workers=max_workers, proxies=proxies, retry=retry, status_forcelist=status_forcelist, timeout=timeout,
                            validate=validate, verify=verify, progress=progress)
            securities = ticker.history(period="ytd", interval="1d", start=None, end=None, adj_timezone=True, adj_ohlc=True)
        except:
            failures.extend(baskets)
            self.failures.update(failures)
            print('[1]', self.failures)
            return
        
        if isinstance(securities, pd.core.frame.DataFrame):
            be_in_memory = set(map(lambda x:x[0], securities.index))
            successes.extend(be_in_memory)
            failures.extend(filter(lambda x: not x in be_in_memory, baskets))
            for security in be_in_memory:
                securities.loc[security].to_csv(os.path.join(self.dataset_dirname, f'{security}.csv'))
        elif isinstance(securities, dict):
            be_in_memory = map(lambda x:x[0], filter(lambda x:not isinstance(x[1], str), zip(securities.keys(), securities.values())))
            not_in_memory = map(lambda x:x[0], filter(lambda x:isinstance(x[1], str), zip(securities.keys(), securities.values())))
            successes.extend(be_in_memory)
            failures.extend(not_in_memory)
            for security in successes:
                fdr.DataReader(security).to_csv(os.path.join(self.dataset_dirname, f'{security}.csv'))

        self.successes.update(successes)
        self.failures.update(failures)
        self._logger_for_successes('from_yahooquery')


    def from_fdr(self, baskets):
        successes = list()
        failures = list()
        for security in tqdm(baskets):
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
