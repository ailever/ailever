import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Queue
import FinanceDataReader as fdr


def _download(bound, queue):
    symbols = pd.read_csv('stockset/KRX.csv').Symbol.values

    exception_list = queue
    for i, symbol in enumerate(tqdm(symbols)):
        if i >= bound[0] and i < bound[1]:
            try:
                if not os.path.isfile(f'stockset/{symbol}.csv'):
                    fdr.DataReader(symbol).to_csv(f'stockset/{symbol}.csv')
            except:
                exception_list.put(symbol)

queue = Queue()
def download(n=30, queue=queue):
    r"""
    Args:
        n:
        queue:
    
    Examples:
        >>> from ailever.forecast.STOCK import krx
        >>> krx.download(n=30)
    """
    if not os.path.isdir('stockset'):
        os.mkdir('stockset')
    
    KIs = ['KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200']
    print('* Korea Composite Stock Price Index Lodaer')
    for KI in tqdm(KIs):
	korea_index = fdr.DataReader(f'{KI}')
	korea_index.to_csv(f'stockset/{KI}.csv')

    krx = fdr.StockListing('KRX')
    if not os.path.isfile(f'stockset/KRX.csv'):
        krx.to_csv('stockset/KRX.csv')

    common_diff = int(len(krx)/int(n))

    print('* Korean Stock Lodaer')
    procs = []
    for _n in range(int(n)):
        lower = int(common_diff * (_n))
        upper = int(common_diff * (_n+1))
        bound = (lower, upper)
        proc = Process(target=_download, args=(bound, queue, ))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
    
    exception_list = []
    for _ in range(queue.qsize()):
        exception_list.append(queue.get())

    return exception_list


def all(date='2010-01-01', mode='Close', cut=None):
    r"""
    Args:
        date:
        mode:
        cut:

    Examples:
        >>> from ailever.forecast.STOCK import krx
        >>> Df = krx.all(date='2010-01-01', mode='Close')
        >>> ...
        >>> stock = Df[0]
        >>> info = Df[1]
        >>> exception_list = Df[2]
    """
    stock_list = pd.read_csv('stockset/KRX.csv').drop('Unnamed: 0', axis=1)
    symbols = stock_list.Symbol.values
    stocks = pd.read_csv('stockset/005930.csv')
    stocks = stocks[stocks.Date >= date][f'{mode}'].values

    # Df[0] & Df[2] : Korean Stock / Exception List
    exception_list = list()
    for i, symbol in enumerate(tqdm(symbols)):
        if i == cut: break
        try:
            stock = pd.read_csv(f'stockset/{symbol}.csv')
            stock = stock[stock.Date >= date][f'{mode}'].values
            if len(stocks) == len(stock):
                stocks = np.c_[stocks, stock]
            else:
                exception_list.append(symbol)
        except:
            exception_list.append(symbol)
    
    # Df[1] : Korean Stock List
    stock_list = stock_list.query(f'Symbol != {exception_list}')

    # Df[3] : Korea Composite Stock Price Index Lodaer
    KIs = ['KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200']
    KI_dict = {}
    for KI in KIs:
	df = pd.read_csv(f'stockset/{KI}.csv')
	KI_dict[KI] = df

    return stocks[:, 1:], stock_list, exception_list, KI_dict, mode


def _all(date='2010-01-01', mode='Close', cut=None):
    date = np.datetime64(date)
    stock_list = fdr.StockListing('KRX')
    symbols = stock_list.Symbol.values
    stocks = fdr.DataReader('005930', date)
    stocks = stocks[f'{mode}'].values

    # Df[0] & Df[2] : Korean Stock / Exception List
    exception_list = list()
    for i, symbol in enumerate(tqdm(symbols)):
        if i == cut: break
        try:
            stock = fdr.DataReader(symbol, date)
            stock = stock[f'{mode}'].values
            stocks = np.c_[stocks, stock]
        except:
            exception_list.append(symbol)
        
    # Df[1] : Korean Stock List
    stock_list = stock_list.query(f'Symbol != {exception_list}')

    # Df[3] : Korea Composite Stock Price Index Lodaer
    KIs = ['KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200']
    KI_dict = {}
    for KI in tqdm(KIs):
	korea_index = fdr.DataReader(f'{KI}')
	KI_dict[KI] = korea_index

    return stocks[:, 1:], stock_list, exception_list, KI_dict, mode


def kospi(date='2010-01-01', mode='Close', cut=None):
    r"""
    Args:
        date:
        mode:
        cut:

    Examples:
        >>> from ailever.forecast.STOCK import krx
        >>> Df = krx.kospi(date='2010-01-01', mode='Close')
        >>> ...
        >>> stock = Df[0]
        >>> info = Df[1]
        >>> exception_list = Df[2]
    """
    stock_list = pd.read_csv('stockset/KRX.csv').drop('Unnamed: 0', axis=1)
    stock_list = stock_list[stock_list.Market == 'KOSPI']
    symbols = stock_list.Symbol.values
    stocks = pd.read_csv('stockset/005930.csv')
    stocks = stocks[stocks.Date >= date][f'{mode}'].values

    # Df[0] & Df[2] : Korean Stock / Exception List
    exception_list = list()
    for i, symbol in enumerate(tqdm(symbols)):
        if i == cut: break
        try:
            stock = pd.read_csv(f'stockset/{symbol}.csv')
            stock = stock[stock.Date >= date][f'{mode}'].values
            if len(stocks) == len(stock):
                stocks = np.c_[stocks, stock]
            else:
                exception_list.append(symbol)
        except:
            exception_list.append(symbol)

    # Df[1] : Korean Stock List
    stock_list = stock_list.query(f'Symbol != {exception_list}')

    # Df[3] : Korea Composite Stock Price Index Lodaer
    KIs = ['KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200']
    KI_dict = {}
    for KI in KIs:
	df = pd.read_csv(f'stockset/{KI}.csv')
	KI_dict[KI] = df

    return stocks[:, 1:], stock_list, exception_list, KI_dict, mode


def _kospi(date='2010-01-01', mode='Close', cut=None):
    date = np.datetime64(date)
    stock_list = fdr.StockListing('KRX')
    stock_list = stock_list[stock_list.Market == 'KOSPI']
    symbols = stock_list.Symbol.values
    stocks = fdr.DataReader('005930', date)
    stocks = stocks[f'{mode}'].values

    # Df[0] & Df[2] : Korean Stock / Exception List
    exception_list = list()
    for i, symbol in enumerate(tqdm(symbols)):
        if i == cut: break
        try:
            stock = fdr.DataReader(symbol, date)
            stock = stock[f'{mode}'].values
            stocks = np.c_[stocks, stock]
        except:
            exception_list.append(symbol)

    # Df[1] : Korean Stock List
    stock_list = stock_list.query(f'Symbol != {exception_list}')

    # Df[3] : Korea Composite Stock Price Index Lodaer
    KIs = ['KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200']
    KI_dict = {}
    for KI in tqdm(KIs):
	korea_index = fdr.DataReader(f'{KI}')
	KI_dict[KI] = korea_index

    return stocks[:, 1:], stock_list, exception_list, KI_dict, mode


def kosdaq(date='2010-01-01', mode='Close', cut=None):
    r"""
    Args:
        date:
        mode:
        cut:

    Examples:
        >>> from ailever.forecast.STOCK import krx
        >>> Df = krx.kosdaq(date='2010-01-01', mode='Close')
        >>> ...
        >>> stock = Df[0]
        >>> info = Df[1]
        >>> exception_list = Df[2]
    """
    stock_list = pd.read_csv('stockset/KRX.csv').drop('Unnamed: 0', axis=1)
    stock_list = stock_list[stock_list.Market == 'KOSDAQ']
    symbols = stock_list.Symbol.values
    stocks = pd.read_csv('stockset/005930.csv')
    stocks = stocks[stocks.Date >= date][f'{mode}'].values

    # Df[0] & Df[2] : Korean Stock / Exception List
    exception_list = list()
    for i, symbol in enumerate(tqdm(symbols)):
        if i == cut: break
        try:
            stock = pd.read_csv(f'stockset/{symbol}.csv')
            stock = stock[stock.Date >= date][f'{mode}'].values
            if len(stocks) == len(stock):
                stocks = np.c_[stocks, stock]
            else:
                exception_list.append(symbol)
        except:
            exception_list.append(symbol)

    # Df[1] : Korean Stock List
    stock_list = stock_list.query(f'Symbol != {exception_list}')

    # Df[3] : Korea Composite Stock Price Index Lodaer
    KIs = ['KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200']
    KI_dict = {}
    for KI in KIs:
	df = pd.read_csv(f'stockset/{KI}.csv')
	KI_dict[KI] = df

    return stocks[:, 1:], stock_list, exception_list, KI_dict, mode


def _kosdaq(date='2010-01-01', mode='Close', cut=None):
    date = np.datetime64(date)
    stock_list = fdr.StockListing('KRX')
    stock_list = stock_list[stock_list.Market == 'KOSDAQ']
    symbols = stock_list.Symbol.values
    stocks = fdr.DataReader('005930', date)
    stocks = stocks[f'{mode}'].values

    # Df[0] & Df[2] : Korean Stock / Exception List
    exception_list = list()
    for i, symbol in enumerate(tqdm(symbols)):
        if i == cut: break
        try:
            stock = fdr.DataReader(symbol, date)
            stock = stock[f'{mode}'].values
            stocks = np.c_[stocks, stock]
        except:
            exception_list.append(symbol)

    # Df[1] : Korean Stock List
    stock_list = stock_list.query(f'Symbol != {exception_list}')

    # Df[3] : Korea Composite Stock Price Index Lodaer
    KIs = ['KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200']
    KI_dict = {}
    for KI in tqdm(KIs):
	korea_index = fdr.DataReader(f'{KI}')
	KI_dict[KI] = korea_index

    return stocks[:, 1:], stock_list, exception_list, KI_dict, mode


def konex(date='2010-01-01', mode='Close', cut=None):
    r"""
    Args:
        date:
        mode:
        cut:

    Examples:
        >>> from ailever.forecast.STOCK import krx
        >>> Df = krx.konex(date='2010-01-01', mode='Close')
        >>> ...
        >>> stock = Df[0]
        >>> info = Df[1]
        >>> exception_list = Df[2]
    """
    stock_list = pd.read_csv('stockset/KRX.csv').drop('Unnamed: 0', axis=1)
    stock_list = stock_list[stock_list.Market == 'KONEX']
    symbols = stock_list.Symbol.values
    stocks = pd.read_csv('stockset/005930.csv')
    stocks = stocks[stocks.Date >= date][f'{mode}'].values

    # Df[0] & Df[2] : Korean Stock / Exception List
    exception_list = list()
    for i, symbol in enumerate(tqdm(symbols)):
        if i == cut: break
        try:
            stock = pd.read_csv(f'stockset/{symbol}.csv')
            stock = stock[stock.Date >= date][f'{mode}'].values
            if len(stocks) == len(stock):
                stocks = np.c_[stocks, stock]
            else:
                exception_list.append(symbol)
        except:
            exception_list.append(symbol)

    # Df[1] : Korean Stock List
    stock_list = stock_list.query(f'Symbol != {exception_list}')

    # Df[3] : Korea Composite Stock Price Index Lodaer
    KIs = ['KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200']
    KI_dict = {}
    for KI in KIs:
	df = pd.read_csv(f'stockset/{KI}.csv')
	KI_dict[KI] = df

    return stocks[:, 1:], stock_list, exception_list, KI_dict, mode


def _konex(date='2010-01-01', mode='Close', cut=None):
    date = np.datetime64(date)
    stock_list = fdr.StockListing('KRX')
    stock_list = stock_list[stock_list.Market == 'KONEX']
    symbols = stock_list.Symbol.values
    stocks = fdr.DataReader('005930', date)
    stocks = stocks[f'{mode}'].values

    # Df[0] & Df[2] : Korean Stock / Exception List
    exception_list = list()
    for i, symbol in enumerate(tqdm(symbols)):
        if i == cut: break
        try:
            stock = fdr.DataReader(symbol, date)
            stock = stock[f'{mode}'].values
            stocks = np.c_[stocks, stock]
        except:
            exception_list.append(symbol)

    # Df[1] : Korean Stock List
    stock_list = stock_list.query(f'Symbol != {exception_list}')

    # Df[3] : Korea Composite Stock Price Index Lodaer
    KIs = ['KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200']
    KI_dict = {}
    for KI in tqdm(KIs):
	korea_index = fdr.DataReader(f'{KI}')
	KI_dict[KI] = korea_index

    return stocks[:, 1:], stock_list, exception_list, KI_dict, mode

