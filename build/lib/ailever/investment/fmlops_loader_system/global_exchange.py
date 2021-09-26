from ailever.investment import __fmlops_bs__ as fmlops_bs
from ..stock_market import MarketInformation
from .._base_transfer import DataTransferCore

from datetime import datetime
import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd

core = fmlops_bs.core['FS1d'] 

def all_exchanges(markets:list):
    # base stock : 005390
    fdr.DataReader('005390').to_csv(os.path.join(core.path, '005390.csv'))
    MI = MarketInformation()
    market_info = MI.market_info[MI.market_info.Market.apply(lambda x: x in markets)].reset_index().drop('index', axis=1)

    def global_exchange(date='2010-01-01', mode='Close', cut=None, baskets=None):
        nonlocal market_info
        # basket filtering
        if baskets:
            origin_baskets = baskets
            
            # filtering
            symbols = market_info.Symbol.values
            baskets = list(filter(lambda x: x in symbols, baskets))
            serialized_objects = list(map(lambda x: x[:-4], core.listfiles(format='csv')))
            baskets = list(filter(lambda x: x in serialized_objects, baskets))
            baskets = np.array(baskets)
        else:
            baskets = market_info.Symbol.values
            origin_baskets = baskets
            
            # filtering
            serialized_objects = list(map(lambda x: x[:-4], core.listfiles(format='csv')))
            baskets = list(filter(lambda x: x in serialized_objects, baskets))
        
        # Df[0] : Price Dataset
        base_stock = pd.read_csv(os.path.join(core.path,'005930.csv'))
        DTC = parallelize(baskets=baskets, path=core.path, base_column=mode, date_column='Date', columns=base_stock.columns.to_list())
        # Df[1] : Stock List
        stock_list = market_info.query(f'Symbol in {baskets}').reset_index().drop('index', axis=1)
        # Df[2] : Exception List
        exception_list = list(filter(lambda x: x not in baskets, origin_baskets))
        # Df[3] : Composite Stock Price Index Lodaer
        financial_market_indicies = dict()
        KIs = ['KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200']
        KI_dict = dict()
        for KI in KIs:
            df = pd.read_csv(os.path.join(core.path, f'{KI}.csv'))
            KI_dict[KI] = df
        financial_market_indicies.update(KI_dict)
        return DTC.ndarray, stock_list, exception_list, financial_market_indicies, mode
    return global_exchange


def parallelize(baskets=None, path=dataset_dirname, object_format='csv', base_column='close', date_column='date', columns=None):
    assert bool(columns), 'The columns argument must be defined'
    if isinstance(columns, pd.core.indexes.base.Index):
        columns = columns.to_list()

    if bool(baskets):
        baskets = list(map(lambda x: x + '.' + object_format, baskets))
        serialized_objects = filter(lambda x: x in baskets, filter(lambda x: x[-len(object_format):] == object_format, os.listdir(path)))
    else:
        serialized_objects = filter(lambda x: x[-len(object_format):] == object_format, os.listdir(path))
    
    base_frame = None
    for so in tqdm(serialized_objects):
        df = pd.read_csv(os.path.join(path, so))
        if df.columns.to_list() == columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)[base_column].to_frame().reset_index()
            df.columns = [date_column, so[:-len(object_format)-1]]
        else:
            continue

        if base_frame is not None:
            base_frame = pd.merge(base_frame, df, on=date_column, how='outer')
        else:
            base_frame = df

    base_frame = base_frame.sort_values(by=date_column).reset_index().drop('index', axis=1).set_index(date_column)
    base_frame.to_csv('.prllz_cache'+'.'+object_format)

    datacore = DataTransferCore()
    datacore.pdframe = base_frame
    datacore.ndarray = base_frame.values
    return datacore
