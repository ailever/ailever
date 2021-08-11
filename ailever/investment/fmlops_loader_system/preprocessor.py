from ailever.investment import fmlops_bs
from ..logger import Logger
from .._base_transfer import DataTransferCore
from .integrated_loader import Loader

import os
from pandas.core.frame import DataFrame

import pandas as pd

base_dir = dict()
base_dir['root'] = fmlops_bs.local_system.root.name
base_dir['rawdata_repository'] = fmlops_bs.local_system.root.rawdata_repository.name
base_dir['preprocessed_repository'] = fmlops_bs.local_system.root.rawdata_repository.preprocessed_repository.name
base_dir['metadata_store'] = fmlops_bs.local_system.root.metadata_store.name
base_dir['feature_store'] = fmlops_bs.local_system.root.feature_store.name
base_dir['model_registry'] = fmlops_bs.local_system.root.model_registry.name
base_dir['source_repotitory'] = fmlops_bs.local_system.root.source_repository.name

logger = Logger()
dataset_dirname = os.path.join(base_dir['root'], base_dir['rawdata_repository'])
preprocessed_dataset_dirname = os.path.join(base_dir['root'], base_dir['rawdata_repository'], base_dir['preprocessed_repository'])
log_dirname = os.path.join(base_dir['root'], base_dir['metadata_store'])


class Preprocessor(DataTransferCore):
    
    overnight = None
    rolling = None

    result = None
    

    def __init__(self, baskets=None, date_column='date', from_dir=dataset_dirname, to_dir=preprocessed_dataset_dirname):

        self.baskets = baskets
        self.from_dir = from_dir
        self.to_dir = to_dir

        self.date_column = date_column 
        self.base_column = [x for x in self.dataframe.columns.to_list() if x != self.date_column]
        self.base_period = self.dataframe[date_column]

    def to_csv(self, **frame):
        pass

    def reset_frame(self):
        pass

    def rounder(self, data, option="round"):
    
        pass

    def missing_values(self, dataframe):
        
        pass
    
    def ohlcv(self, baskets=None, from_dir=None, to_dir=None, window=None, ticker=None, merge=None, output="pdframe"):
        pass


    def pct_change(self, baskets=None, from_dir=None, to_dir=None, window=None, ticker=None, merge=None, output="pdframe"):
        
        r"""---------- Initializing args ----------"""
        if not from_dir:
           from_dir = self.from_dir
        if not to_dir:
            to_dir = self.to_dir
        if not baskets:
            serialized_objects = os.listdir(from_dir)
            serialized_object =list(filter(lambda x: x[-3:] == 'csv', serialized_objects))
            baskets_in_dir = list(map(lambda x: x[:-4], serialized_object))
            baskets = baskets_in_dir
            logger.normal_logger.info(f"[PREPROCESSOR] NO BASKETS INPUT: All the Baskets from {from_dir}")
        logger.normal_logger.info(f"[PREPROCSSEOR] ACCESS TO LOADER FOR {baskets} UPDATE")
        loader = Loader()
        frame = loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=from_dir) 
        pdframe = frame.dict
        return

    def overnight(self, baskets=None, from_dir=None, to_dir=None, ticker=None, merge=None, output="pdframe"):
        pass
        """df = self.dataframe
        df_cross = pd.concat([df.open, df.close.shift()], axis=1)
        df_overnight = df_cross.assign(overnight=lambda x: (x['open']/x['close']) -1)['overnight']
        
        self.result = pd.concat([df, df_overnight], axis=1)
        
        if output == "pdframe":
            self.overnight = pd.concat([df[self.date_column], df_overnight], axis=1)
        if output == "ndarray":
            self.overnight = df_overnight.to_numpy()

        return self"""

    def rolling(self, column="close", window=10, rolling_type="mean", ticker=None, merge=None, output="pdframe"):
        pass
        """df = self.dataframe
        df_rolling = df[column].rolling(window=window,
                                min_periods=None,
                                center=False,
                                win_type=None,
                                axis=0,
                                closed=None)
        if rolling_type =="mean":
            new_column = column + "_" + str(window) + rolling_type
            self.result = pd.concat([df, df_rolling.mean().rename(new_column, inplace=True)], axis=1)             
            
            if output =="pdframe":
                self.rolling = pd.concat([df[self.date_column], df_rolling.mean().rename(new_column, inplace=True)], axis=1)            
            if output =="ndarray":
                self.rolling = df_rolling.mean().to_numpy()

        if rolling_type =="sum":
            new_column = column + "_" + str(window) + rolling_type
            self.result = pd.concat([df, df_rolling.sum().rename(new_column, inplace=True)], axis=1)             

            if output =="pdframe":
                self.rolling = pd.concat([df[self.date_column], df_rolling.sum().rename(new_column, inplace=True)], axis=1)            
            if output =="ndarray":
                self.rolling = df_rolling.sum().to_numpy()
            
        return(self)"""

    def relative(self):
        pass


    def stochastic(self):
        pass
