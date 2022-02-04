## Portfolio Optimization
### Composing Portfolio
`FinanceDataReader`
```python
import FinanceDataReader as fdr

tickers = ['TSLA', 'FB']
histories = list()
for ticker in tickers:
    histories.append(fdr.DataReader(ticker))

portfolio = pd.concat(histories, join='outer', axis=1).asfreq('B').fillna(method='bfill')
portfolio.columns = pd.MultiIndex.from_product([tickers, ['Close', 'Open', 'High', 'Low', 'Volume', 'Change']])
portfolio = portfolio.swaplevel(i=0, j=1, axis=1)[['Close', 'Open', 'High', 'Low', 'Volume', 'Change']]
portfolio
```

`yahooquery`
```python
from yahooquery import Ticker

tickers = Ticker(['ARE', 'FB'])
portfolio = tickers.history(start='2010-01-01').asfreq('B').fillna(method='bfill')
portfolio
```

`pandasdatareader`
```python
from pandas_datareader import data

portfolio = data.DataReader(['TSLA', 'FB'], 'yahoo', start='2010/01/01', end='2019/12/31').asfreq('B').fillna(method='bfill')
portfolio
```

### Expected Portfolio Returns
```python
```

### Portfolio Volatility
```python
```

