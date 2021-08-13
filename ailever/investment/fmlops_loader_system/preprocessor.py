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
    def to_csv(self, to_dir=None):
        if not self.dict:
            logger.normal_logger.info('[PREPROCESSOR] NO FRAME TO CONVERT INTO CSV. PLEASE CHECK self.dict or self.preprocessed_list')
            return
        if not to_dir:
            to_dir=self.to_dir
        baskets = list(self.dict.keys()) 
        for ticker in baskets:
            csv_file_name = ticker+'_'+('_'.join(self.preprocessed_list))+'.csv'
            self.dict[ticker].to_csv(os.path.join(to_dir, csv_file_name), index=False)
        logger.normal_logger.info(f'[PREPROCESSOR] TICKER WITH {self.preprocessed_list} OUTPUT TO CSV')

    def reset(self):
        
        self.preprocessed_list = list()
        self.dict = dict()
        self.merged = False
        logger.normal_logger.info(f'[PREPROCESSOR] FRAME HAS BEEN RESET - {self.preprocessed_list} Cleared')

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

    def missing_values(self, dataframe):
        
        pass

    def pct_change(self, baskets=None, from_dir=None, to_dir=None, target_column=None, window=None, merge=None, kind=False):
        
        r"""---------- Initializing args ----------"""
        if not kind:
            logger.normal_logger.info(f"[PREPROCESSOR] NO KIND INPUT. DECIDE ON ticker or index_full or index_single")
            return
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
        if kind =="ticker":  
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
                ohlcv_ticker_pdframe = all_frame[ticker]
                date_column_pdframe = ohlcv_ticker_pdframe[['date']]
                pct_change_list = list()
                for w in window:    
                    pct_change_single = ohlcv_ticker_pdframe[target_column].pct_change(periods=w).to_frame()
                    pct_change_list.append(pct_change_single)
                pct_change_pdframe = pd.concat(pct_change_list, axis=1)
                pct_change_pdframe.columns = pct_change_column_list
                if merge:
                    if not self.merged:
                        ticker_pdframe = pd.concat([ohlcv_ticker_pdframe, pct_change_pdframe], axis=1)
                    if self.merged: 
                        ticker_pdframe = pd.concat([self.dict[ticker], pct_change_pdframe], axis=1)
                if not merge:
                    ticker_pdframe = pd.concat([date_column_pdframe, pct_change_pdframe], axis=1)
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

        if 'index' in kind:
            if not self.dict:
                logger.normal_logger.info("[PREPROCESSOR] NO BASKETS TO ATTACH INDEXES TO")
                return
            if not baskets:
                logger.normal_logger.info("[PREPROCESSOR] NO INDEXES INPUT")
                return
            logger.normal_logger.info(f"[PREPROCSSEOR] ACCESS TO LOADER FOR {baskets} UPDATE")
            """Initializing loader for data updates"""
            loader = Loader()
            index_frame = loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=from_dir).dict
            index_dict = dict()
            index_preprocessed = list()
            for index in baskets:
                pct_change_column_list = [ index+'+'+target_column+'+change'+str(w) for w in window ]
                ohlcv_index_pdframe = index_frame[index]
                date_column_pdframe = ohlcv_index_pdframe[['date']]
                pct_change_list = list()
                for w in window:    
                    pct_change_single = ohlcv_index_pdframe[target_column].pct_change(periods=w).to_frame()
                    pct_change_list.append(pct_change_single)
                pct_change_pdframe = pd.concat(pct_change_list, axis=1)
                pct_change_pdframe.columns = pct_change_column_list
                if kind == "index_full":
                    ohlcv_index_pdframe.columns = [ 'date' ,index+'open', index+'high', index+'low', index+'close', index+'volume' ]
                    index_pdframe = pd.concat([ohlcv_index_pdframe, pct_change_pdframe], axis=1)
                    index_preprocessed.extend([index+'ohlcv'])
                    index_preprocessed.extend(pct_change_column_list)
                if kind == "index_single":
                    index_pdframe = pd.concat([date_column_pdframe, pct_change_pdframe], axis=1)
                    index_preprocessed.extend(pct_change_column_list)
                index_dict[index] = index_pdframe
            
            merged_dict = dict()
            ticker_dict = self.dict
            for ticker in list(ticker_dict.keys()):
                merged_frame = ticker_dict[ticker]
                for index in list(index_dict.keys()):
                    merged_frame = merged_frame.merge(index_dict[index], on='date', how='outer')
                merged_dict[ticker] = merged_frame
            self.merged= True
            self.preprocessed_list.extend(index_preprocessed)
            self.dict = merged_dict
            logger.normal_logger.info(f'[PREPROCESSOR] {index_preprocessed} MERGED TO BASKETS')
            return self

    def overnight(self, baskets=None, from_dir=None, to_dir=None, ticker=None, merge=None, output="pdframe", ticker_on=False):
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

    def rolling(self, column="close", window=10, rolling_type="mean", ticker=None, merge=None, output="pdframe", ticker_on=False):
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
