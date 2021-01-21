import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Process, Queue
import FinanceDataReader as fdr



def _download(bound, queue):
    if not os.path.isdir('stockset'):
        os.mkdir('stockset')

    if not os.path.isfile(f'stockset/KRX.csv'):
        krx = fdr.StockListing('KRX')
        krx.to_csv('stockset/KRX.csv')

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
def download(n=100, queue=queue):
    krx = fdr.StockListing('KRX')
    common_diff = int(len(krx)/int(n))

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
    stock_list = pd.read_csv('stockset/KRX.csv').drop('Unnamed: 0', axis=1)
    symbols = stock_list.Symbol.values
    stocks = pd.read_csv('stockset/005930.csv')
    stocks = stocks[stocks.Date >= date][f'{mode}'].values

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
    
    stock_list = stock_list.query(f'Symbol != {exception_list}')
    return stocks[:, 1:], stock_list, exception_list


def _all(date='2010-01-01', mode='Close', cut=None):
    date = np.datetime64(date)
    stock_list = fdr.StockListing('KRX')
    symbols = stock_list.Symbol.values
    stocks = fdr.DataReader('005930', date)
    stocks = stocks[f'{mode}'].values

    exception_list = list()
    for i, symbol in enumerate(tqdm(symbols)):
        if i == cut: break
        try:
            stock = fdr.DataReader(symbol, date)
            stock = stock[f'{mode}'].values
            stocks = np.c_[stocks, stock]
        except:
            exception_list.append(symbol)
        
    stock_list = stock_list.query(f'Symbol != {exception_list}')
    return stocks[:, 1:], stock_list, exception_list


def kospi(date='2010-01-01', mode='Close', cut=None):
    stock_list = pd.read_csv('stockset/KRX.csv').drop('Unnamed: 0', axis=1)
    stock_list = stock_list[stock_list.Market == 'KOSPI']
    symbols = stock_list.Symbol.values
    stocks = pd.read_csv('stockset/005930.csv')
    stocks = stocks[stocks.Date >= date][f'{mode}'].values

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

    stock_list = stock_list.query(f'Symbol != {exception_list}')
    return stocks[:, 1:], stock_list, exception_list


def _kospi(date='2010-01-01', mode='Close', cut=None):
    date = np.datetime64(date)
    stock_list = fdr.StockListing('KRX')
    stock_list = stock_list[stock_list.Market == 'KOSPI']
    symbols = stock_list.Symbol.values
    stocks = fdr.DataReader('005930', date)
    stocks = stocks[f'{mode}'].values

    exception_list = list()
    for i, symbol in enumerate(tqdm(symbols)):
        if i == cut: break
        try:
            stock = fdr.DataReader(symbol, date)
            stock = stock[f'{mode}'].values
            stocks = np.c_[stocks, stock]
        except:
            exception_list.append(symbol)

    stock_list = stock_list.query(f'Symbol != {exception_list}')
    return stocks[:, 1:], stock_list, exception_list


def kosdaq(date='2010-01-01', mode='Close', cut=None):
    stock_list = pd.read_csv('stockset/KRX.csv').drop('Unnamed: 0', axis=1)
    stock_list = stock_list[stock_list.Market == 'KOSDAQ']
    symbols = stock_list.Symbol.values
    stocks = pd.read_csv('stockset/005930.csv')
    stocks = stocks[stocks.Date >= date][f'{mode}'].values

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

    stock_list = stock_list.query(f'Symbol != {exception_list}')
    return stocks[:, 1:], stock_list, exception_list


def _kosdaq(date='2010-01-01', mode='Close', cut=None):
    date = np.datetime64(date)
    stock_list = fdr.StockListing('KRX')
    stock_list = stock_list[stock_list.Market == 'KOSDAQ']
    symbols = stock_list.Symbol.values
    stocks = fdr.DataReader('005930', date)
    stocks = stocks[f'{mode}'].values

    exception_list = list()
    for i, symbol in enumerate(tqdm(symbols)):
        if i == cut: break
        try:
            stock = fdr.DataReader(symbol, date)
            stock = stock[f'{mode}'].values
            stocks = np.c_[stocks, stock]
        except:
            exception_list.append(symbol)

    stock_list = stock_list.query(f'Symbol != {exception_list}')
    return stocks[:, 1:], stock_list, exception_list


def konex(date='2010-01-01', mode='Close', cut=None):
    stock_list = pd.read_csv('stockset/KRX.csv').drop('Unnamed: 0', axis=1)
    stock_list = stock_list[stock_list.Market == 'KONEX']
    symbols = stock_list.Symbol.values
    stocks = pd.read_csv('stockset/005930.csv')
    stocks = stocks[stocks.Date >= date][f'{mode}'].values

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

    stock_list = stock_list.query(f'Symbol != {exception_list}')
    return stocks[:, 1:], stock_list, exception_list


def _konex(date='2010-01-01', mode='Close', cut=None):
    date = np.datetime64(date)
    stock_list = fdr.StockListing('KRX')
    stock_list = stock_list[stock_list.Market == 'KONEX']
    symbols = stock_list.Symbol.values
    stocks = fdr.DataReader('005930', date)
    stocks = stocks[f'{mode}'].values

    exception_list = list()
    for i, symbol in enumerate(tqdm(symbols)):
        if i == cut: break
        try:
            stock = fdr.DataReader(symbol, date)
            stock = stock[f'{mode}'].values
            stocks = np.c_[stocks, stock]
        except:
            exception_list.append(symbol)

    stock_list = stock_list.query(f'Symbol != {exception_list}')
    return stocks[:, 1:], stock_list, exception_list

