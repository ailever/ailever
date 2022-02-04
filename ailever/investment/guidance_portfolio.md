## Portfolio Optimization
``
```python
import FinanceDataReader as fdr

tickers = ['TSLA', 'FB']
histories = list()
for ticker in tickers:
    histories.append(fdr.DataReader(ticker))

portfolio = pd.concat(histories, join='outer', axis=1).fillna(method='bfill')
portfolio.columns = pd.MultiIndex.from_product([tickers, ['Close', 'Open', 'High', 'Low', 'Volume', 'Change']])
portfolio = portfolio.swaplevel(i=0, j=1, axis=1)[['Close', 'Open', 'High', 'Low', 'Volume', 'Change']]
portfolio
```

`yahooquery`
```python

```

`pandasdatareader`
```python

```
