from ..path import refine

from datetime import datetime
import os
import re
import numpy as np
import pandas as pd

def parallelize(path='.', object_format='csv', base_column='close', date_column='date', period=100):
    prllz = Parallelizer(path=path,
                         object_format=object_format,
                         base_column=base_column,
                         date_column=date_column,
                         truncate=period)
    return prllz

class Parallelizer:
    def __init__(self, path, object_format, base_column, date_column, truncate):
        self.origin_path = os.getcwd()
        self.serialization_path = refine(path)
        self.base_column = base_column
        self.date_column = date_column
        self.truncated_period = truncate
        self.ndarray = getattr(self, '_'+object_format)(to='ndarray')
        self.pdframe = getattr(self, '_'+object_format)(to='pdframe')
        
    def _csv(self, to):
        serialized_objects = os.listdir(self.serialization_path)
        serialized_objects = list(filter(lambda x: x[-3:] == 'csv', serialized_objects))
        ticker_names = list(map(lambda x: x[:-re.search('[.]', x[::-1]).span()[1]], serialized_objects))
        
        so_path = os.path.join(self.serialization_path, serialized_objects.pop(0))
        base_init_frame = pd.read_csv(so_path)
        base_period = base_init_frame[self.date_column].values[-self.truncated_period:]
        base_array = base_init_frame[self.base_column].values[-self.truncated_period:]

        mismatching = list()
        for so in serialized_objects:
            so_path = os.path.join(self.serialization_path, so)
            appending_frame = pd.read_csv(so_path)
            appending_period = appending_frame[self.date_column].values[-self.truncated_period:]
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
            print('[AILEVER] Mismatched tickers for given period :', mismatching)
            return base_array

        elif to == 'pdframe':
            base_frame = pd.DataFrame(data=base_array, columns=ticker_names)
            base_frame.insert(0, self.date_column, pd.to_datetime(base_period))
            return base_frame.set_index(self.date_column)

    

