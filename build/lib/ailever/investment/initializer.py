import os
import pandas as pd
import FinanceDataReader as fdr

def initialize():
    if not os.path.isfile('KRX.csv'):
        fdr.StockListing('KRX').to_csv('KRX.csv')
    elif not os.path.isfile('KOSPI.csv'):
        fdr.StockListing('KOSPI').to_csv('KOSPI.csv')
    elif not os.path.isfile('KOSDAQ.csv'):
        fdr.StockListing('KOSDAQ').to_csv('KOSDAQ.csv')
    elif not os.path.isfile('KONEX.csv'):
        fdr.StockListing('KONEX').to_csv('KONEX.csv')
    elif not os.path.isfile('NYSE.csv'):
        fdr.StockListing('NYSE').to_csv('NYSE.csv')
    elif not os.path.isfile('NASDAQ.csv'):
        fdr.StockListing('NASDAQ').to_csv('NASDAQ.csv')
    elif not os.path.isfile('AMEX.csv'):
        fdr.StockListing('AMEX').to_csv('AMEX.csv')
    elif not os.path.isfile('S&P500.csv'):
        fdr.StockListing('S&P500').to_csv('S&P500.csv')
    elif not os.path.isfile('SSE.csv'):
        fdr.StockListing('SSE').to_csv('SSE.csv')
    elif not os.path.isfile('SZSE.csv'):
        fdr.StockListing('SZSE').to_csv('SZSE.csv')
    elif not os.path.isfile('HKEX.csv'):
        fdr.StockListing('HKEX').to_csv('HKEX.csv')
    elif not os.path.isfile('TSE.csv'):
        fdr.StockListing('TSE').to_csv('TSE.csv')
    elif not os.path.isfile('HOSE.csv'):
        fdr.StockListing('HOSE').to_csv('HOSE.csv')
    elif not os.path.isfile('KRX-DELISTING.csv'):
        fdr.StockListing('KRX-DELISTING').to_csv('KRX-DELISTING.csv')
    elif not os.path.isfile('KRX-ADMINISTRATIVE.csv'):
        fdr.StockListing('KRX-ADMINISTRATIVE').to_csv('KRX-ADMINISTRATIVE.csv')

    stocks = pd.read_csv('KRX.csv').drop('Unnamed: 0',axis=1)[['Market', 'Symbol', 'Name', 'Industry']].sort_values(by='Market')
    for market in ['NYSE', 'NASDAQ', 'AMEX', 'HKEX', 'HOSE', 'S&P500', 'SSE', 'TSE']:
        df = pd.read_csv(f'{market}.csv').drop('Unnamed: 0', axis=1)[['Symbol', 'Name', 'Industry']]
        df['Market'] = market
        df = df[['Market', 'Symbol', 'Name', 'Industry']]
        stocks = stocks.append(df).reset_index().drop('index', axis=1)
    stocks = stocks.drop_duplicates()
    return stocks
