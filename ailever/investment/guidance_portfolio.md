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

ticker_names = ['TSLA', 'FB']
tickers = Ticker(ticker_names)
histories = tickers.history(start='2010-01-01')

portfolio = pd.concat([histories.loc[ticker_name] for ticker_name in histories.index.get_level_values(0).unique()], join='outer', axis=1).asfreq('B').fillna(method='bfill')
portfolio.columns = pd.MultiIndex.from_product([ticker_names, histories.loc[histories.index.get_level_values(0).unique()[0]].columns.tolist()])
portfolio = portfolio.swaplevel(i=0, j=1, axis=1)[histories.loc[histories.index.get_level_values(0).unique()[0]].columns.tolist()]
portfolio
```

`pandasdatareader`
```python
from pandas_datareader import data

portfolio = data.DataReader(['TSLA', 'FB'], 'yahoo', start='2010/01/01', end='2019/12/31').asfreq('B').fillna(method='bfill')
portfolio
```

### Expected Security Returns and Volatility
```python
from yahooquery import Ticker

ticker_names = ['TSLA', 'FB']

tickers = Ticker(ticker_names)
histories = tickers.history(start='2010-01-01')

portfolio = pd.concat([histories.loc[ticker_name] for ticker_name in histories.index.get_level_values(0).unique()], join='outer', axis=1).asfreq('B').fillna(method='bfill')
portfolio.columns = pd.MultiIndex.from_product([ticker_names, histories.loc[histories.index.get_level_values(0).unique()[0]].columns.tolist()])
portfolio = portfolio.swaplevel(i=0, j=1, axis=1)[histories.loc[histories.index.get_level_values(0).unique()[0]].columns.tolist()]

expected_returns = portfolio['adjclose'].pct_change().fillna(0).applymap(lambda x: np.log(1+x)).mean()
daily_volatility = portfolio['adjclose'].pct_change().fillna(0).applymap(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(1))
annual_volatility = portfolio['adjclose'].pct_change().fillna(0).applymap(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

assets = pd.concat([expected_returns, daily_volatility, annual_volatility], axis=1)
assets.columns = ['Returns', 'DailyVolatility', 'AnnualVolatility']
assets
```

### Expected Portfolio Returns and Volatility
```python
```

### Expected Weighted Portfolio Returns and Volatility
```python
from yahooquery import Ticker

ticker_names = ['TSLA', 'FB']
ticker_weights = [0.2, 0.8]

tickers = Ticker(ticker_names)
histories = tickers.history(start='2010-01-01')

portfolio = pd.concat([histories.loc[ticker_name] for ticker_name in histories.index.get_level_values(0).unique()], join='outer', axis=1).asfreq('B').fillna(method='bfill')
portfolio.columns = pd.MultiIndex.from_product([ticker_names, histories.loc[histories.index.get_level_values(0).unique()[0]].columns.tolist()])
portfolio = portfolio.swaplevel(i=0, j=1, axis=1)[histories.loc[histories.index.get_level_values(0).unique()[0]].columns.tolist()]

security_means = portfolio['adjclose'].pct_change().fillna(0).applymap(lambda x: np.log(1+x)).mean()
security_variance = portfolio['adjclose'].pct_change().fillna(0).applymap(lambda x: np.log(1+x)).cov()

expected_returns = security_means.mul(ticker_weights).sum()
daily_volatility = np.sqrt(1)*np.sqrt(security_variance.mul(ticker_weights, axis=0).mul(ticker_weights, axis=1).sum().sum())
annual_volatility = np.sqrt(252)*np.sqrt(security_variance.mul(ticker_weights, axis=0).mul(ticker_weights, axis=1).sum().sum())
expected_returns, daily_volatility, annual_volatility
```

### Efficient Frontier

```python

```

