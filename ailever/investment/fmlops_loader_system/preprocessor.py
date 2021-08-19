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

    def __init__(self, baskets=None, from_dir=preprocessed_dataset_dirname, to_dir=preprocessed_dataset_dirname, interval='1d'):

        self.baskets = baskets
        self.from_dir = from_dir
        self.to_dir = to_dir
        self.interval = interval
        self.dict = dict()
        self.merged = False

        self.preprocessed_list = list()


    def to_csv(self, to_dir=None, interval=None, option=None):
        if not self.dict:
            logger.normal_logger.info('[PREPROCESSOR] NO FRAME TO CONVERT INTO CSV. PLEASE CHECK self.dict or self.preprocessed_list')
            return
        if not to_dir:
            to_dir=self.to_dir
        baskets = list(self.dict.keys())
        if not interval:
            interval = self.interval
        for ticker in baskets:
            csv_file_name = ticker+'_'+('_'.join(self.preprocessed_list))+'.csv'
            if option == 'dropna':
                self.dict[ticker].dropna().to_csv(os.path.join(to_dir, csv_file_name), index=True)
            if option != 'dropna':
                self.dict[ticker].to_csv(os.path.join(to_dir, csv_file_name), index=True)
        logger.normal_logger.info(f'[PREPROCESSOR] TICKER WITH {self.preprocessed_list} OUTPUT TO CSV')

    def na_handler(self, option=None):
        if not option:
            option = 'dropna'
        frame = self.dict
        for ticker in list(self.dict.key()):
            frame[ticker] = frame[ticker].dropna()
        self.dict = frame
        return self

    def reset(self, target_column=None):
        
        if not target_column:
            self.preprocessed_list = list()
            self.dict = dict()
            self.merged = False
            logger.normal_logger.info(f'[PREPROCESSOR] FRAME HAS BEEN RESET - {self.preprocessed_list} Cleared')
        if isinstance(target_column, int):
            self.preprocessed_list = list[:-1]
            for ticker in list(self.dict.keys()):
                self.dict[ticker] = self.dict[ticker].iloc[:,:-target_column]
        if isinstance(target_column, list):
            for c in target_column:
                self.preprocessed_list.remove(c)
            for ticker in list(self.dict.keys()):
                self.dict[ticker] = self.dict[ticker].drop(target_column, axis=1)
        if isinstance(target_column, str):
            self.preprocessed_list.remove(target_column)
            for ticker in list(self.dict.keys()):
                self.dict[ticker] = self.dict[ticker].drop(target_column, axis=1)
        
        return self

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

    def date_featuring(self):
        date_index = pd.to_datetime(self.dict[list(self.dict.keys())[0]].index.to_series())
        date_featured = pd.concat([date_index.apply(lambda x: x.year), date_index.apply(lambda x: x.quarter), date_index.apply(lambda x: x.month), date_index.apply(lambda x: x.day), date_index.apply(lambda x: x.dayofweek)], axis=1)
        date_featured.columns = ['year', 'quarter', 'month', 'day', 'dayofweek']
        for ticker in list(self.dict.keys()):
            merged_frame = date_featured.merge(self.dict[ticker], how='outer', left_index=True, right_index=True, suffixes=('L', 'R'))
            self.dict[ticker] = merged_frame
        logger.normal_logger.info('[PREPROCESSOR] DATE FEATURED YEAR, MONTH, DAY, DAY OF WEEK ADDED')
        return self 

    def ohlcv(self, baskets=None, from_dir=None, to_dir=None, window=None, ticker=None, merge=None):
        
        pass

    def missing_values(self, dataframe):
        
        pass

    def _pct_change_base(self, baskets=None, from_dir=None, to_dir=None, interval=None, country='united states', target_column=None, window=None, merge=None):
        logger.normal_logger.info(f"[PREPROCSSEOR] ACCESS TO LOADER FOR {baskets} UPDATE")
        #Initializing loader for data updates
        loader = Loader()
        frame = loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=from_dir, interval=interval, country=country) 
        all_frame = frame.dict
        pct_change_column_list = [ target_column+'+change'+str(w) for w in window ]
        for ticker in baskets:
            ohlcv_ticker_pdframe = all_frame[ticker].reset_index()
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
                    try:
                        ticker_pdframe = pd.concat([self.dict[ticker].reset_index(), pct_change_pdframe], axis=1)
                    except KeyError:
                        logger.normal_logger.info('TICKERS ARE NOT MATCHED: Previous Tickers {pre} vs Current Baskets: {baskets}. Try Reset FRAME'.format(pre=list(self.dict.keys()), baskets=baskets))
            if not merge:
                ticker_pdframe = pd.concat([date_column_pdframe, pct_change_pdframe], axis=1)
            self.dict[ticker] = ticker_pdframe.set_index('date')
        if merge:
            logger.normal_logger.info(f'[PREPROCESSOR] {pct_change_column_list} MERGED')
            self.merged = True
            self.preprocessed_list.extend(pct_change_column_list)
            if not self.ohlcv_name in self.preprocessed_list:
                self.preprocessed_list.insert(0, self.ohlcv_name)
        if not merge:
            logger.normal_logger.info(f'[PREPROCESSOR] {pct_change_column_list} SINGLE PDFRAME')
            self.merged = False
            self.preprocessed_list = list()
            self.preprocessed_list.extend(pct_change_column_list)
        return self
    
    def _pct_change_index(self, from_dir=None, to_dir=None, interval=None, country='united states', target_column=None, window=None, merge=None, financial_index=None, index_type='full'):
        if not self.dict:
            logger.normal_logger.info("[PREPROCESSOR] NO BASKETS TO ATTACH INDEXES TO")
            return

        baskets = financial_index
        logger.normal_logger.info(f"[PREPROCSSEOR] ACCESS TO LOADER FOR {baskets} UPDATE")
        """Initializing loader for data updates"""
        loader = Loader()
        index_frame = loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=from_dir, interval=interval, country=country).dict
        index_dict = dict()
        index_preprocessed = list()
        for f_index in financial_index:
            pct_change_column_list = [ f_index+'+'+target_column+'+change'+str(w) for w in window ]
            ohlcv_index_pdframe = index_frame[f_index].reset_index()
            date_column_pdframe = ohlcv_index_pdframe[['date']]
            pct_change_list = list()
            for w in window:    
                pct_change_single = ohlcv_index_pdframe[target_column].pct_change(periods=w).to_frame()
                pct_change_list.append(pct_change_single)
            pct_change_pdframe = pd.concat(pct_change_list, axis=1)
            pct_change_pdframe.columns = pct_change_column_list
            if index_type == "full":
                ohlcv_index_pdframe.columns = [ 'date' ,f_index+'open', f_index+'high', f_index+'low', f_index+'close', f_index+'volume' ]
                index_pdframe = pd.concat([ohlcv_index_pdframe, pct_change_pdframe], axis=1)
                index_pdframe.set_index('date', inplace=True)
                index_preprocessed.extend([f_index+'+'+self.ohlcv_name])
                index_preprocessed.extend(pct_change_column_list)
            if index_type == "single":
                index_pdframe = pd.concat([date_column_pdframe, pct_change_pdframe], axis=1)
                index_pdframe.set_index('date', inplace=True)
                index_preprocessed.extend(pct_change_column_list)
            index_dict[f_index] = index_pdframe
        
        merged_dict = dict()
        ticker_dict = self.dict
        for ticker in list(ticker_dict.keys()):
            merged_frame = ticker_dict[ticker]
            for index in list(index_dict.keys()):
                merged_frame = merged_frame.merge(index_dict[index], how='outer', left_index=True, right_index=True, suffixes=('L', 'R'))
            merged_dict[ticker] = merged_frame
        self.merged= True
        self.preprocessed_list.extend(index_preprocessed)
        self.dict = merged_dict
        logger.normal_logger.info(f'[PREPROCESSOR] {index_preprocessed} MERGED TO BASKETS')

    def pct_change(self, baskets=None, from_dir=None, to_dir=None, interval=None, country='united states', target_column=None, window=None, merge=None, financial_index=None, index_type='full'):
        
        r"""---------- Initializing args ----------"""
        if not from_dir:
           from_dir = self.from_dir
        logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT FROM_DIR - {from_dir}")
        if not to_dir:
            to_dir = self.to_dir
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT TO_DIR - {to_dir}")
        if not interval:
            interval = self.interval
        if not target_column:
            target_column = 'close'
            logger.normal_logger.info(f'[PREPROCESSOR] DEFAULT TARGET_COLUMN - {target_column}')
        if type(window)==str or type(window)==int:
            logger.noral_logger.info(f'[PREPROCESSOR] WINDOW INPUT MUST BE IN LIST')
            return
        if not window:
            window = [1,5,20]
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT WINDOW FOR PCT_CHANGE - {window}")
        if not merge:
            merge = True
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT MERGE OPTION TRUE")
        self.ohlcv_name = f'ohlcv+{interval}'

        if not financial_index:
            self = self._pct_change_base(baskets=baskets, from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, target_column=target_column, window=window, merge=merge)
        else:
            if baskets:
                self = self._pct_change_base(baskets=baskets, from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, target_column=target_column, window=window, merge=merge)
                self = self._pct_change_index(from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, target_column=target_column, window=window, merge=merge, financial_index=financial_index, index_type=index_type)
            else:
                self = self._pct_change_index(from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, target_column=target_column, window=window, merge=merge, financial_index=financial_index, index_type=index_type)
        return self

    def _overnight_base(self, baskets=None, from_dir=None, to_dir=None, interval=None, country='united states', merge=None):
        logger.normal_logger.info(f"[PREPROCSSEOR] ACCESS TO LOADER FOR {baskets} UPDATE")
        """Initializing loader for data updates"""
        loader = Loader()
        frame = loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=from_dir, interval=interval, country=country) 
        all_frame = frame.dict 
        ohlcv_overnight_column = 'overnight'
        for ticker in baskets:
            ohlcv_ticker_pdframe = all_frame[ticker].reset_index()
            date_column_pdframe = ohlcv_ticker_pdframe[['date']]
            ohlcv_ticker_pdframe_cross = pd.concat([ohlcv_ticker_pdframe.open, ohlcv_ticker_pdframe.close.shift()], axis=1)
            ohlcv_ticker_pdframe_overnight = ohlcv_ticker_pdframe_cross.assign(overnight=lambda x: (x['open']/x['close']) -1)['overnight'].to_frame()
            if merge:
                if not self.merged:
                    ticker_pdframe = pd.concat([ohlcv_ticker_pdframe, ohlcv_ticker_pdframe_overnight], axis=1)
                if self.merged: 
                    try:
                        ticker_pdframe = pd.concat([self.dict[ticker].reset_index(), ohlcv_ticker_pdframe_overnight], axis=1)
                    except KeyError:
                        logger.normal_logger.info('TICKERS ARE NOT MATCHED: Previous Tickers {pre} vs Current Baskets: {baskets}. Try Reset FRAME'.format(pre=list(self.dict.keys()), baskets=baskets))
            if not merge:
                ticker_pdframe = pd.concat([date_column_pdframe, ohlcv_ticker_pdframe_overnight], axis=1)
            self.dict[ticker] = ticker_pdframe.set_index('date')
        if merge:
            logger.normal_logger.info(f'[PREPROCESSOR] {ohlcv_overnight_column} MERGED')
            self.merged = True
            self.preprocessed_list.extend([ohlcv_overnight_column])
            if not self.ohlcv_name in self.preprocessed_list:
                self.preprocessed_list.insert(0, self.ohlcv_name)
        if not merge:
            logger.normal_logger.info(f'[PREPROCESSOR] {ohlcv_overnight_column} SINGLE PDFRAME')
            self.merged = False
            self.preprocessed_list = list()
            self.preprocessed_list.extend([ohlcv_overnight_column])
        return self

    def _overnight_index(self, from_dir=None, to_dir=None, interval=None, country='united_states', merge=None, financial_index=None, index_type='full'):
        if not self.dict:
            logger.normal_logger.info("[PREPROCESSOR] NO BASKETS TO ATTACH INDEXES TO")
            return

        baskets = financial_index
        logger.normal_logger.info(f"[PREPROCSSEOR] ACCESS TO LOADER FOR {baskets} UPDATE")
        """Initializing loader for data updates"""
        loader = Loader()
        index_frame = loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=from_dir, interval=interval, country=country).dict
        index_dict = dict()
        index_preprocessed = list()
        for index in baskets:
            ohlcv_overnight_column = index+'+overnight'
            ohlcv_index_pdframe = index_frame[index].reset_index()
            date_column_pdframe = ohlcv_index_pdframe[['date']]
            ohlcv_index_pdframe_cross = pd.concat([ohlcv_index_pdframe.open, ohlcv_index_pdframe.close.shift()], axis=1)
            ohlcv_index_pdframe_overnight = ohlcv_index_pdframe_cross.assign(overnight=lambda x: (x['open']/x['close']) -1)['overnight'].to_frame()
            ohlcv_index_pdframe_overnight.columns = [index+'overnight']
            if index_type == "full":
                ohlcv_index_pdframe.columns = [ 'date' ,index+'open', index+'high', index+'low', index+'close', index+'volume' ]
                index_pdframe = pd.concat([ohlcv_index_pdframe, ohlcv_index_pdframe_overnight], axis=1)
                index_pdframe.set_index('date', inplace=True)
                index_preprocessed.extend([index+'+'+self.ohlcv_name])
                index_preprocessed.extend([ohlcv_overnight_column])
            if index_type == "single":
                index_pdframe = pd.concat([date_column_pdframe, ohlcv_index_pdframe_overnight], axis=1)
                index_pdframe.set_index('date', inplace=True)
                index_preprocessed.extend([ohlcv_overnight_column])
            index_dict[index] = index_pdframe
        
        merged_dict = dict()
        ticker_dict = self.dict
        for ticker in list(ticker_dict.keys()):
            merged_frame = ticker_dict[ticker]
            for index in list(index_dict.keys()):
                merged_frame = merged_frame.merge(index_dict[index],how='outer', left_index=True, right_index=True, suffixes=('L', 'R'))
            merged_dict[ticker] = merged_frame
        self.merged= True
        self.preprocessed_list.extend(index_preprocessed)
        self.dict = merged_dict
        logger.normal_logger.info(f'[PREPROCESSOR] {index_preprocessed} MERGED TO BASKETS')
        return self

    def overnight(self, baskets=None, from_dir=None, to_dir=None, interval=None, country='united states', merge=None, financial_index=None, index_type='full'):
        r"""---------- Initializing args ----------"""
        if not from_dir:
           from_dir = self.from_dir
        logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT FROM_DIR - {from_dir}")
        if not to_dir:
            to_dir = self.to_dir
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT TO_DIR - {to_dir}")
        if not interval:
            interval = self.interval
        if not merge:
            merge = True
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT MERGE OPTION TRUE")
        self.ohlcv_name = f'ohlcv+{interval}'

        if not financial_index:
            self = self._overnight_base(baskets=baskets, from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, merge=merge)
        else:
            if baskets:
                self = self._overnight_base(baskets=baskets, from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, merge=merge)
                self = self._overnight_index(from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, merge=merge, financial_index=financial_index, index_type=index_type)
            else:
                self = self._overnight_index(from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, merge=merge, financial_index=financial_index, index_type=index_type)
        return self




    def _rolling_base(self, baskets=None, from_dir=None, to_dir=None, interval=None, country='united states', target_column=None, window=None, win_type=None, rolling_type=None, merge=None):

        logger.normal_logger.info(f"[PREPROCSSEOR] ACCESS TO LOADER FOR {baskets} UPDATE")
        """Initializing loader for data updates"""
        loader = Loader()
        frame = loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=from_dir, interval=interval, country=country) 
        all_frame = frame.dict
        rolling_column_list = [target_column+'+rolling('+rolling_type+')'+str(w) for w in window ]
        for ticker in baskets:
            ohlcv_ticker_pdframe = all_frame[ticker].reset_index()
            date_column_pdframe = ohlcv_ticker_pdframe[['date']]
            rolling_list = list()
            for w in window:    
                rolling_single = getattr(ohlcv_ticker_pdframe[target_column].rolling(window=w,
                                                                            min_periods=None,
                                                                            center=False,
                                                                            win_type=win_type,
                                                                            axis=0,
                                                                            closed=None), rolling_type)().to_frame()

                rolling_list.append(rolling_single)
            rolling_pdframe = pd.concat(rolling_list, axis=1)
            rolling_pdframe.columns = rolling_column_list
            if merge:
                if not self.merged:
                    ticker_pdframe = pd.concat([ohlcv_ticker_pdframe, rolling_pdframe], axis=1)
                if self.merged: 
                    try:
                        ticker_pdframe = pd.concat([self.dict[ticker].reset_index(), rolling_pdframe], axis=1)
                    except KeyError:
                        logger.normal_logger.info('TICKERS ARE NOT MATCHED: Previous Tickers {pre} vs Current Baskets: {baskets}. Try Reset FRAME'.format(pre=list(self.dict.keys()), baskets=baskets))
            if not merge:
                ticker_pdframe = pd.concat([date_column_pdframe, rolling_pdframe], axis=1)
            self.dict[ticker] = ticker_pdframe.set_index('date')
        if merge:
            logger.normal_logger.info(f'[PREPROCESSOR] {rolling_column_list} MERGED')
            self.merged = True
            self.preprocessed_list.extend(rolling_column_list)
            if not self.ohlcv_name in self.preprocessed_list:
                self.preprocessed_list.insert(0, self.ohlcv_name)
        if not merge:
            logger.normal_logger.info(f'[PREPROCESSOR] {rolling_column_list} SINGLE PDFRAME')
            self.merged = False
            self.preprocessed_list = list()
            self.preprocessed_list.extend(rolling_column_list)
        return self

    def _rolling_index(self, from_dir=None, to_dir=None, interval=None, country='united states', target_column=None, window=None, win_type=None, rolling_type=None, merge=None, financial_index=None, index_type='full'):
        if not self.dict:
            logger.normal_logger.info("[PREPROCESSOR] NO BASKETS TO ATTACH INDEXES TO")
            return

        baskets = financial_index
        logger.normal_logger.info(f"[PREPROCSSEOR] ACCESS TO LOADER FOR {baskets} UPDATE")
        """Initializing loader for data updates"""
        loader = Loader()
        index_frame = loader.ohlcv_loader(baskets=baskets, from_dir=from_dir, to_dir=from_dir, interval=interval, country=country).dict
        index_dict = dict()
        index_preprocessed = list()
        for index in baskets:
            rolling_column_list = [ index+'+'+target_column+'+rolling('+rolling_type+')'+str(w) for w in window ]
            ohlcv_index_pdframe = index_frame[index].reset_index()
            date_column_pdframe = ohlcv_index_pdframe[['date']]
            rolling_list = list()
            for w in window:    
                rolling_single = getattr(ohlcv_index_pdframe[target_column].rolling(window=w,
                                                                            min_periods=None,
                                                                            center=False,
                                                                            win_type=win_type,
                                                                            axis=0,
                                                                            closed=None), rolling_type)().to_frame()


                rolling_list.append(rolling_single)
            rolling_pdframe = pd.concat(rolling_list, axis=1)
            rolling_pdframe.columns = rolling_column_list
            if index_type == "full":
                ohlcv_index_pdframe.columns = [ 'date' ,index+'open', index+'high', index+'low', index+'close', index+'volume' ]
                index_pdframe = pd.concat([ohlcv_index_pdframe, rolling_pdframe], axis=1)
                index_pdframe.set_index('date', inplace=True)
                index_preprocessed.extend([index+'+'+self.ohlcv_name])
                index_preprocessed.extend(rolling_column_list)
            if index_type == "single":
                index_pdframe = pd.concat([date_column_pdframe, rolling_pdframe], axis=1)
                index_pdframe.set_index('date', inplace=True)
                index_preprocessed.extend(rolling_column_list)
            index_dict[index] = index_pdframe
        
        merged_dict = dict()
        ticker_dict = self.dict
        for ticker in list(ticker_dict.keys()):
            merged_frame = ticker_dict[ticker]
            for index in list(index_dict.keys()):
                merged_frame = merged_frame.merge(index_dict[index], how='outer', left_index=True, right_index=True, suffixes=('L', 'R'))
            merged_dict[ticker] = merged_frame
        self.merged= True
        self.preprocessed_list.extend(index_preprocessed)
        self.dict = merged_dict
        logger.normal_logger.info(f'[PREPROCESSOR] {index_preprocessed} MERGED TO BASKETS')
        return self

    def rolling(self, baskets=None, from_dir=None, to_dir=None, interval=None, country='united states', target_column=None, window=None, win_type=None, rolling_type=None, merge=None, financial_index=None, index_type='full'):
            
        r"""---------- Initializing args ----------"""
        if not from_dir:
           from_dir = self.from_dir
        logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT FROM_DIR - {from_dir}")
        if not to_dir:
            to_dir = self.to_dir
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT TO_DIR - {to_dir}")
        if not interval:
            interval = self.interval
        if not target_column:
            target_column = 'close'
            logger.normal_logger.info(f'[PREPROCESSOR] DEFAULT TARGET_COLUMN - {target_column}')
        if type(window)==str or type(window)==int:
            logger.noral_logger.info(f'[PREPROCESSOR] WINDOW INPUT MUST BE IN LIST')
            return
        if not window:
            window = [5,20, 60]
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT WINDOW FOR PCT_CHANGE - {window}")
        if not rolling_type:
            rolling_type = 'mean'
        if not merge:
            merge = True
            logger.normal_logger.info(f"[PREPROCESSOR] DEFAULT MERGE OPTION TRUE")
        self.ohlcv_name = f'ohlcv+{interval}'

        if not financial_index:
            self = self._rolling_base(baskets=baskets, from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, target_column=target_column, window=window, win_type=win_type, rolling_type=rolling_type, merge=merge)
        else:
            if baskets:
                self = self._rolling_base(baskets=baskets, from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, target_column=target_column, window=window, win_type=win_type, rolling_type=rolling_type, merge=merge)
                self = self._rolling_index(from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, target_column=target_column, window=window, win_type=win_type, rolling_type=rolling_type, merge=merge, financial_index=financial_index, index_type=index_type)
            else:
                self = self._rolling_index(from_dir=from_dir, to_dir=to_dir, interval=interval, country=country, target_column=target_column, window=window, win_type=win_type, rolling_type=rolling_type, merge=merge, financial_index=financial_index, index_type=index_type)
        return self



    def relative(self):
        pass


    def stochastic(self):
        pass
