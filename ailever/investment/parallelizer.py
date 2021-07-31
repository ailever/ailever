from ..path import refine

from datetime import datetime
import os
import re
import numpy as np
import pandas as pd

def parallelize(path='.', object_format='csv'):
    prllz = Parallelizer(path=refine(path), object_format=object_format)
    return prllz.ndarray

class Parallelizer:
    def __init__(self, path, object_format, truncate=100):
        self.origin_path = os.getcwd()
        self.serialization_path = path
        self.truncated_period = truncate
        self.ndarray = getattr(self, '_'+object_format)(to='ndarray')
        self.pdframe = getattr(self, '_'+object_format)(to='pdframe')
        
    def _csv(self, to):
        serialized_objects = os.listdir(self.serialization_path)
        ticker_names = list(map(lambda x: x[-re.search('[.]', x[::-1]).span()[1]:], serialized_objects))
        
        so_path = os.path.join(self.serialization_path, serialized_objects.pop(0))
        base = pd.read_csv(so_path)['close'].values[-self.truncated_period:]
        for so in serialized_objects:
            so_path = os.path.join(self.serialization_path, so)
            base = np.c_[base, pd.read_csv(so_path)['close'].values[-self.truncated_period:]]

        if to == 'ndarray':
            return base
        elif to == 'pdframe':
            base = pd.DataFrame(data=base, columns=ticker_names)
            base.insert(0, 'Date', pd.date_range(end=datetime.today().date(), freq='d', periods=self.truncated_period))
            return base.set_index('Date')

    
