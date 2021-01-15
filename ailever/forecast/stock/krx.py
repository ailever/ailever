import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import FinanceDataReader as fdr


def download():
    if not os.path.isdir('stockset'):
        os.mkdir('stockset')
        fdr.StockListing('KRX').to_csv('stockset/KRX.csv')
        symbols = pd.read_csv('stockset/KRX.csv').Symbol.values

        for symbol in tqdm(symbols):
            fdr.DataReader(symbol).to_csv(f'stockset/{symbol}.csv')



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

    return stocks[:, 1:], stock_list, exception_list

