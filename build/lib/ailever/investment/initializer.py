import FinanceDataReader as fdr

def initialize():
    fdr.StockListing('KRX').to_csv('KRX.csv')
    fdr.StockListing('KOSPI').to_csv('KOSPI.csv')
    fdr.StockListing('KOSDAQ').to_csv('KOSDAQ.csv')
    fdr.StockListing('KONEX').to_csv('KONEX.csv')
    fdr.StockListing('NYSE').to_csv('NYSE.csv')
    fdr.StockListing('NASDAQ').to_csv('NASDAQ.csv')
    fdr.StockListing('AMEX').to_csv('AMEX.csv')
    fdr.StockListing('S&P500').to_csv('S&P500.csv')
    fdr.StockListing('SSE').to_csv('SSE.csv')
    fdr.StockListing('SZSE').to_csv('SZSE.csv')
    fdr.StockListing('HKEX').to_csv('HKEX.csv')
    fdr.StockListing('TSE').to_csv('TSE.csv')
    fdr.StockListing('HOSE').to_csv('HOSE.csv')
    fdr.StockListing('KRX-DELISTING').to_csv('KRX-DELISTING.csv')
    fdr.StockListing('KRX-ADMINISTRATIVE').to_csv('KRX-ADMINISTRATIVE.csv')
