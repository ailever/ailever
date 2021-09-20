from ailever.investment import __fmlops_bs__ as fmlops_bs
from .._base_transfer import DataTransferCore
from .integrated_loader import Loader

from datetime import datetime
import os
import re
import numpy as np
import pandas as pd

dataset_dirname = fmlops_bs.core['FS'].path

class Parallelization_Loader:
    def __init__(self):
        self.loader = Loader()

    def prllz_loader(self, baskets=None, path=dataset_dirname, object_format='csv', base_column='close', date_column='date', period=100):
        self.loader.ohlcv_loader(baskets=baskets)
        self.prllz = Parallelizer(baskets=baskets,
                             path=path,
                             object_format=object_format,
                             base_column=base_column,
                             date_column=date_column,
                             truncate=period)
        self.datacore = DataTransferCore()
        self.datacore.ndarray = self.prllz.ndarray
        self.datacore.pdframe = self.prllz.pdframe
        return self.datacore

    @staticmethod
    def parallelize(baskets=None, path=dataset_dirname, object_format='csv', base_column='close', date_column='date', period=100):
        prllz = Parallelizer(baskets=baskets,
                             path=path,
                             object_format=object_format,
                             base_column=base_column,
                             date_column=date_column,
                             truncate=period)
        datacore = DataTransferCore()
        datacore.ndarray = prllz.ndarray
        datacore.pdframe = prllz.pdframe
        return datacore


class Parallelizer:
    def __init__(self, baskets, object_format, base_column, date_column, truncate, path=dataset_dirname):
        self.baskets = baskets
        self.origin_path = os.getcwd()
        self.serialization_path = path
        self.base_column = base_column
        self.date_column = date_column
        self.truncated_period = truncate
        self.ndarray = getattr(self, '_'+object_format)(to='ndarray')
        self.pdframe = getattr(self, '_'+object_format)(to='pdframe')
 
    def _csv(self, to):
        if not self.baskets:
            serialized_objects = [file for file in os.listdir(self.serialization_path) if os.path.isfile(os.path.join(self.serialization_path, file))]
        else:
            serialized_objects = list(map(lambda x: x+'.csv', self.baskets))
        serialized_objects = list(filter(lambda x: (x[-3:] == 'csv') and ('_' not in x) and ('+' not in x), serialized_objects))
        ticker_names = list(map(lambda x: x[:-re.search('[.]', x[::-1]).span()[1]], serialized_objects))
        
        so_path = os.path.join(self.serialization_path, serialized_objects.pop(0))
        base_init_frame = pd.read_csv(so_path)
        base_period = base_init_frame[self.date_column].values[-self.truncated_period:]
        self.base_period = base_period
        base_array = base_init_frame[self.base_column].values[-self.truncated_period:]
        mismatching = list()
        for so in serialized_objects:
            so_path = os.path.join(self.serialization_path, so)
            appending_frame = pd.read_csv(so_path)
            try:
                appending_period = appending_frame[self.date_column].values[-self.truncated_period:]
            except:
                print(f'FAIL TO LOAD : {so}')
                continue
            checker = base_period == appending_period
            if not checker.all():
                so = so[:-re.search('[.]', so[::-1]).span()[1]]
                mismatching.append(so)
                continue
            appending_array = appending_frame[self.base_column].values[-self.truncated_period:]
            base_array = np.c_[base_array, appending_array]
        
        for m_ticker in mismatching:
            del ticker_names[ticker_names.index(m_ticker)]

        if to == 'ndarray':
            print('[PARALLELIZER] MISMATCHED TiCKERS FOR GIVEN PERIOD :', mismatching)
            return base_array

        elif to == 'pdframe':
            base_frame = pd.DataFrame(data=base_array, columns=ticker_names)
            base_frame.insert(0, self.date_column, pd.to_datetime(base_period))
            return base_frame.set_index(self.date_column)

    
