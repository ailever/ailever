from ailever.investment import __fmlops_bs__ as fmlops_bs
from ..logger import Logger
from .._base_transfer import DataTransferCore
from .integrated_loader import Loader

from pandas.core.frame import DataFrame

import math
import os
import pandas as pd
from collections import OrderedDict


base_dir = dict()
base_dir['root'] = fmlops_bs.local_system.root.name
base_dir['metadata_store'] = fmlops_bs.local_system.root.metadata_store.name
base_dir['feature_store'] = fmlops_bs.local_system.root.feature_store.name
base_dir['model_registry'] = fmlops_bs.local_system.root.model_registry.name
base_dir['source_repotitory'] = fmlops_bs.local_system.root.source_repository.name

logger = Logger()
preprocessed_dataset_dirname = os.path.join(base_dir['root'], base_dir['feature_store'])
log_dirname = os.path.join(base_dir['root'], base_dir['metadata_store'])

# ARE_OHLCV_F1_F2_TickerOpen.csv

"""ARE_ohlcv_rolling10_overnight_VIXovernight_US10YXTovernight.csv
O
BXP"""



class Preprocessor(DataTransferCore):

    def __init__(self, baskets=None, from_dir=preprocessed_dataset_dirname, to_dir=preprocessed_dataset_dirname):

        self.baskets = baskets
        self.from_dir = from_dir
        self.to_dir = to_dir
        self.dict = dict()
        self.merged = False

        self.preprocessed_list = list()
        r""" {ticker:ohlcv, ticker:ticker_high, ticker:ticker_low, ticker:ticker_close, ticker:ticker_volume, ticker:ticekr_rolling, index:index_overnight}"""

    def to_csv(self, to_dir):
        baskets = list(self.dict.keys()) 
        for ticker in baskets:
            pd.to_csv(f'{ticker}.csv')
        columns = list(self.preprocessed_list.keys())
        logger.normal_logger.info(f'[PREPROCESSOR] {columns} OUTPUT TO CSV - {baskets}')

    def reset_frame(self):
        
        self.preprocessed_dict = OrderedDict()
        self.dict = dict()

    def rounder(self, data, option="round", digit=4):
        
        if round=='round':
            return round(data, digit)
        if round=='ceiling':
            temp = data * (10^digit)
            temp_converted = math.ceiling(temp)
            refined_data = temp_converted/(10^digit)
            return refined_data
        if round=='floor':
            temp = data * (10^digit)
            temp_converted = math.ceiling(temp)
            refined_data = temp_converted/(10^digit)
            return refined_data


    def ohlcv(self, baskets=None, from_dir=None, to_dir=None, window=None, ticker=None, merge=None):
        
        pass

    def index_preprocessor(self, baskets=None, indexes=None, from_dir=None, to_dir=None, functions=None, target_column=None, window=None, merge=None ,ticker=True):
        
        r"""---------- Initializing args ----------"""
        if not from_dir:
           from_dir = self.from_dir
        logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT FROM_DIR - {from_dir}")
        if not to_dir:
            to_dir = self.to_dir
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT TO_DIR - {to_dir}")
        if not target_column:
            target_column = 'close'
            logger.normal_logger.info(f'[PREPROCESSOR] DEFAULT TARGET_COLUMN - {target_column}')
        if type(window)==str:
            window = list(window)
        if not window:
            window = [1,5,20]
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT WINDOW FOR PCT_CHANGE - {window}")
        if not merge:
            merge = True
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT MERGE OPTION TRUE")
        if not baskets:
            serialized_objects = os.listdir(from_dir)
            serialized_object =list(filter(lambda x: x[-3:] == 'csv', serialized_objects))
            baskets_in_dir = list(map(lambda x: x[:-4], serialized_object))
            baskets = baskets_in_dir
            logger.normal_logger.info(f"[PREPROCESSOR] NO BASKETS INPUT: All the Baskets from {from_dir}")
        if not indexes:
            index = ['VIX']
            logger.normal_logger.info(f"[PREPROCESSOR] NO INDEX INPUT - Default Index {index}")

        logger.normal_logger.info(f"[PREPROCSSEOR] ACCESS TO LOADER FOR {baskets} UPDATE")
        """Initializing loader for baskets updates"""
        loader = Loader()
        frame = loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=from_dir) 
        all_frame = frame.dict
        """Initializing loader for index updates"""
        logger.normal_logger.info(f"[PREPROCSSEOR] ACCESS TO LOADER FOR {index} UPDATE")
        loader_index = Loader()
        frame = loader_index.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=from_dir) 
        all_frame_index = frame_index.dict
        index_processor = Preprocessor()
        return


    def missing_values(self, dataframe):
        
        pass

    def pct_change(self, baskets=None, from_dir=None, to_dir=None, target_column=None, window=None, merge=None):
        
        r"""---------- Initializing args ----------"""
        if not from_dir:
           from_dir = self.from_dir
        logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT FROM_DIR - {from_dir}")
        if not to_dir:
            to_dir = self.to_dir
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT TO_DIR - {to_dir}")
        if not target_column:
            target_column = 'close'
            logger.normal_logger.info(f'[PREPROCESSOR] DEFAULT TARGET_COLUMN - {target_column}')
        if type(window)==str:
            window = list(window)
        if not window:
            window = [1,5,20]
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT WINDOW FOR PCT_CHANGE - {window}")
        if not merge:
            merge = True
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT MERGE OPTION TRUE")
        if not baskets:
            serialized_objects = os.listdir(from_dir)
            serialized_object =list(filter(lambda x: x[-3:] == 'csv', serialized_objects))
            baskets_in_dir = list(map(lambda x: x[:-4], serialized_object))
            baskets = baskets_in_dir
            logger.normal_logger.info(f"[PREPROCESSOR] NO BASKETS INPUT: All the Baskets from {from_dir}")
        logger.normal_logger.info(f"[PREPROCSSEOR] ACCESS TO LOADER FOR {baskets} UPDATE")
        """Initializing loader for data updates"""
        loader = Loader()
        frame = loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=from_dir) 
        all_frame = frame.dict
        pct_change_column_list = [ target_column+'+change'+str(w) for w in window ]
        for ticker in baskets:
            ohlcv_ticker_pdframe = all_frame[ticker].reset_index()
            date_column_pdframe = ohlcv_ticker_pdframe[['date']]
            '''pct_change_list = list(all_pdframe[ticker]['date'])'''
            pct_change_list = list()
            '''pct_change_column_list = ['date']'''
            for w in window:    
                pct_change_single = ohlcv_ticker_pdframe[target_column].pct_change(periods=w).to_frame()
                file_name = f'{ticker}_{target_column}+change{w}.csv' 
                pd.concat([date_column_pdframe, pct_change_single], axis=1).to_csv(os.path.join(to_dir, file_name), index=False)
                pct_change_list.append(pct_change_single)
            pct_change_pdframe = pd.concat(pct_change_list, axis=1)
            pct_change_pdframe.columns = pct_change_column_list
            if merge:
                if not self.merged:
                    '''ticker_pdframe = pd.merge(all_pdframe[ticker], pct_change_frame, on='date', how='outer')'''
                    ticker_pdframe = pd.concat([ohlcv_ticker_pdframe, pct_change_pdframe], axis=1)
                    '''self.merged = True'''
                if self.merged: 
                    ticker_pdframe = pd.concat([self.dict[ticker], pct_change_pdframe], axis=1)
            if not merge:
                ticker_pdframe = pd.concat([date_column_pdframe, pct_change_pdframe], axis=1)
                '''self.merged = False'''
            self.dict[ticker] = ticker_pdframe
        if merge:
            logger.normal_logger.info(f'[PREPROCESSOR] {pct_change_column_list} MERGED')
            self.merged = True
            self.preprocessed_list.extend(pct_change_column_list)
            if not 'ohlcv' in self.preprocessed_list:
                self.preprocessed_list.insert(0, 'ohlcv')
        if not merge:
            logger.normal_logger.info(f'[PREPROCESSOR] {pct_change_column_list} SINGLE PDFRAME')
            self.merged = False
            self.preprocessed_list = list()
            self.preprocessed_list.extend(pct_change_column_list)
        return self

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
