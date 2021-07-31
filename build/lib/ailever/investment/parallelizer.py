from ..path import refile

import os
import re
import numpy as np
import pandas as pd


def parallelize(path='.', object_format='csv'):
    prllz = Parallelizer(path=refine(path), object_format=object_format)
    return prllz.ndarray

class Parallelizer:
    def __init__(self, path, object_format):
        self.origin_path = os.getcwd()
        self.temporary_path = path
        setattr(self, ndarray, getattr(self, '_'+object_format))
        
    def _csv(self):
        os.chdir()
        serialized_objects = os.listdir()
        ticker_names = map(lambda x: x[-re.search('[.]', x[::-1]).span()[1]:], serialized_objects)
        
        base = pd.read_csv(serialized_objects.pop(0))['close'].values[-100:]
        for so in serialized_objects:
            base = np.c_[base, pd.read_csv(so)['close'].values[-100:]]
        return base

    

