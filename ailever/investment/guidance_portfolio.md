- [Guidance: Analysis](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_analysis.md)
- [Guidance: Dataset](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_dataset.md)
- [Guidance: Model](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_model.md)
- [Guidance: Management](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_management.md)
- [Guidance: Portfolio](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_portfolio.md)

---

## Portfolio Optimization
- https://en.wikipedia.org/wiki/Modern_portfolio_theory
- https://en.wikipedia.org/wiki/Power_transform
- https://en.wikipedia.org/wiki/Volatility_(finance)

### Composing Portfolio
`FinanceDataReader`
```python
import FinanceDataReader as fdr

ticker_names = ['TSLA', 'FB']
histories = list()
for ticker in ticker_names:
    histories.append(fdr.DataReader(ticker))

portfolio = pd.concat(histories, join='outer', axis=1).asfreq('B').fillna(method='bfill')
portfolio.columns = pd.MultiIndex.from_product([ticker_names, histories[0].columns.tolist()])
portfolio = portfolio.swaplevel(i=0, j=1, axis=1)[histories[0].columns.tolist()]
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

