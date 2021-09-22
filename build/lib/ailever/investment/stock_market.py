import os
import pandas as pd
import FinanceDataReader as fdr

sources = pd.DataFrame(columns=['Name', 'Url'],
                       data=[['finviz', 'https://finviz.com/'],
                             ['investing','https://www.investing.com/'],
                             ['fred', 'https://fred.stlouisfed.org/']])

def market_information(baskets=None, only_symbol=False, inverse_mapping=False, source=False):
    r"""
    TARGET FRAME:
    Code	Name	Market	Dept	Close	ChangeCode	Changes	ChagesRatio	Open	High	Low	Volume	Amount	Marcap	Stocks
    """
    if source:
        return sources

    MI = MarketInformation()
    if baskets:
        market_info = MI.market_query(baskets=baskets, only_symbol=only_symbol, inverse_mapping=inverse_mapping)
    else:
        market_info = MI.market_info
    return market_info

class MarketInformation:
    def __init__(self):
        self.market_info = self.market_information()

    def market_query(self, baskets:list, only_symbol:bool=False, inverse_mapping=False):
        if not isinstance(baskets, list):
            baskets = [baskets]

        if not inverse_mapping:
            market_info = self.market_info.set_index('Name').loc[baskets].reset_index()
            if only_symbol:
                market_info = market_info.Symbol.to_list()
        else:
            market_info = self.market_info.set_index('Symbol').loc[baskets].reset_index()
            market_info = market_info.Name.to_list()

        return market_info

    def market_information(self):
        if not os.path.isfile('KRX.csv'):
            df = fdr.StockListing('KRX')
            df['Region'] = 'Korea'
            df.to_csv('KRX.csv')
        if not os.path.isfile('KOSPI.csv'):
            df = fdr.StockListing('KOSPI')
            df['Region'] = 'Korea'
            df.to_csv('KOSPI.csv')
        if not os.path.isfile('KOSDAQ.csv'):
            df = fdr.StockListing('KOSDAQ')
            df['Region'] = 'Korea'
            df.to_csv('KOSDAQ.csv')
        if not os.path.isfile('KONEX.csv'):
            df = fdr.StockListing('KONEX')
            df['Region'] = 'Korea'
            df.to_csv('KONEX.csv')
        if not os.path.isfile('NYSE.csv'):
            df = fdr.StockListing('NYSE')
            df['Region'] = 'United States'
            df.to_csv('NYSE.csv')
        if not os.path.isfile('NASDAQ.csv'):
            df = fdr.StockListing('NASDAQ')
            df['Region'] = 'United States'
            df.to_csv('NASDAQ.csv')
        if not os.path.isfile('AMEX.csv'):
            df = fdr.StockListing('AMEX')
            df['Region'] = 'United States'
            df.to_csv('AMEX.csv')
        if not os.path.isfile('S&P500.csv'):
            df = fdr.StockListing('S&P500')
            df['Region'] = 'United States'
            df.to_csv('S&P500.csv')
        if not os.path.isfile('SSE.csv'):
            df = fdr.StockListing('SSE')
            df['Region'] = 'China'
            df.to_csv('SSE.csv')
        if not os.path.isfile('SZSE.csv'):
            df = fdr.StockListing('SZSE')
            df['Region'] = 'China'
            df.to_csv('SZSE.csv')
        if not os.path.isfile('HKEX.csv'):
            df = fdr.StockListing('HKEX')
            df['Region'] = 'Hong Kong'
            df.to_csv('HKEX.csv')
        if not os.path.isfile('TSE.csv'):
            df = fdr.StockListing('TSE')
            df['Region'] = 'Japan'
            df.to_csv('TSE.csv')
        if not os.path.isfile('HOSE.csv'):
            df = fdr.StockListing('HOSE')
            df['Region'] = 'Vietnam'
            df.to_csv('HOSE.csv')
        if not os.path.isfile('KRX-DELISTING.csv'):
            df = fdr.StockListing('KRX-DELISTING')
            df['Region'] = 'Korea'
            df.to_csv('KRX-DELISTING.csv')
        if not os.path.isfile('KRX-ADMINISTRATIVE.csv'):
            df = fdr.StockListing('KRX-ADMINISTRATIVE')
            df['Region'] = 'Korea'
            df.to_csv('KRX-ADMINISTRATIVE.csv')

        stocks = pd.read_csv('KRX.csv').drop('Unnamed: 0',axis=1)[['Market', 'Symbol', 'Name', 'Industry', 'Region']].sort_values(by='Market')
        for market in ['NYSE', 'NASDAQ', 'AMEX', 'HKEX', 'HOSE', 'S&P500', 'SSE', 'TSE']:
            df = pd.read_csv(f'{market}.csv').drop('Unnamed: 0', axis=1)[['Symbol', 'Name', 'Industry', 'Region']]
            df['Market'] = market
            df = df[['Market', 'Symbol', 'Name', 'Industry', 'Region']]
            stocks = stocks.append(df).reset_index().drop('index', axis=1)
        stocks = stocks.drop_duplicates()
        return stocks
