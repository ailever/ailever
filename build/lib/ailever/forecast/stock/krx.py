import numpy as np
import FinanceDataReader as fdr

def all(date='2010-01-01', mode='Close'):
    stock_list = fdr.StockListing('KRX')
    stock_list = stock_list[stock_list.ListingDate <= np.datetime64(date)]
    symbols = stock_list.Symbol

    for i, symbol in enumerate(symbols):
        if i == 0:
            stocks = fdr.DataReader(symbol)[f'{mode}'].values
        else:
            stock = fdr.DataReader(symbol)[f'{mode}'].values
            stocks = np.c_[stocks, stock]

    return stocks, stock_list


def kospi(date='2010-01-01'):
    stock_list = fdr.StockListing('KRX')
    stock_list = stock_list[stock_list.Market == 'KOSPI']
    stock_list = stock_list[stock_list.ListingDate <= np.datetime64(date)]
    symbols = stock_list.Symbol

    for i, symbol in enumerate(symbols):
        if i == 0:
            stocks = fdr.DataReader(symbol)[f'{mode}'].values
        else:
            stock = fdr.DataReader(symbol)[f'{mode}'].values
            stocks = np.c_[stocks, stock]

    return stocks, stock_list


def kosdaq(date='2010-01-01'):
    stock_list = fdr.StockListing('KRX')
    stock_list = stock_list[stock_list.Market == 'KOSDAQ']
    stock_list = stock_list[stock_list.ListingDate <= np.datetime64(date)]
    symbols = stock_list.Symbol

    for i, symbol in enumerate(symbols):
        if i == 0:
            stocks = fdr.DataReader(symbol)[f'{mode}'].values
        else:
            stock = fdr.DataReader(symbol)[f'{mode}'].values
            stocks = np.c_[stocks, stock]

    return stocks, stock_list


def konex(date='2010-01-01'):
    stock_list = fdr.StockListing('KRX')
    stock_list = stock_list[stock_list.Market == 'KONEX']
    stock_list = stock_list[stock_list.ListingDate <= np.datetime64(date)]
    symbols = stock_list.Symbol

    for i, symbol in enumerate(symbols):
        if i == 0:
            stocks = fdr.DataReader(symbol)[f'{mode}'].values
        else:
            stock = fdr.DataReader(symbol)[f'{mode}'].values
            stocks = np.c_[stocks, stock]

    return stocks, stock_list
