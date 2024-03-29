## Navigation
- [Guidance: Analysis](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_analysis.md)
- [Guidance: Dataset](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_dataset.md)
- [Guidance: Model](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_model.md)
- [Guidance: Management](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_management.md)
- [Guidance: StockValuation](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_stock_valuation.md)

---

## Portfolio Optimization

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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yahooquery import Ticker

ticker_names = ['TSLA', 'FB']
ticker_weight = [0.2, 0.8]

tickers = Ticker(ticker_names)
histories = tickers.history(start='2010-01-01')

portfolio = pd.concat([histories.loc[ticker_name] for ticker_name in histories.index.get_level_values(0).unique()], join='outer', axis=1).asfreq('B').fillna(method='bfill')
portfolio.columns = pd.MultiIndex.from_product([ticker_names, histories.loc[histories.index.get_level_values(0).unique()[0]].columns.tolist()])
portfolio = portfolio.swaplevel(i=0, j=1, axis=1)[histories.loc[histories.index.get_level_values(0).unique()[0]].columns.tolist()]

security_means = portfolio['adjclose'].pct_change().fillna(0).applymap(lambda x: np.log(1+x)).mean()
security_variance = portfolio['adjclose'].pct_change().fillna(0).applymap(lambda x: np.log(1+x)).cov()

expected_returns = security_means.mul(ticker_weight).sum()
daily_volatility = np.sqrt(1)*np.sqrt(security_variance.mul(ticker_weight, axis=0).mul(ticker_weight, axis=1).sum().sum())
annual_volatility = np.sqrt(252)*np.sqrt(security_variance.mul(ticker_weight, axis=0).mul(ticker_weight, axis=1).sum().sum())
expected_returns, daily_volatility, annual_volatility
```

### Efficient Frontier

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from yahooquery import Ticker

ticker_names = ['TSLA', 'FB', 'ARE']
ticker_weights = np.random.uniform(size=(1000,len(ticker_names)))
ticker_weights = np.r_[ticker_weights, np.array([[10, 9, 4]])]
ticker_weights = ticker_weights/ticker_weights.sum(axis=1)[:,np.newaxis]

tickers = Ticker(ticker_names)
histories = tickers.history(start='2010-01-01')

portfolio = pd.concat([histories.loc[ticker_name] for ticker_name in histories.index.get_level_values(0).unique()], join='outer', axis=1)
portfolio.index = pd.to_datetime(portfolio.index)
portfolio = portfolio.sort_index().asfreq('B').fillna(method='bfill')
portfolio.columns = pd.MultiIndex.from_product([ticker_names, histories.loc[histories.index.get_level_values(0).unique()[0]].columns.tolist()])
portfolio = portfolio.swaplevel(i=0, j=1, axis=1)[histories.loc[histories.index.get_level_values(0).unique()[0]].columns.tolist()]

security_means = portfolio['adjclose'].pct_change().fillna(0).applymap(lambda x: np.log(1+x)).mean()
security_variance = portfolio['adjclose'].pct_change().fillna(0).applymap(lambda x: np.log(1+x)).cov()

portfolios = list()
for ticker_weight in ticker_weights:
    expected_returns = security_means.mul(ticker_weight).sum()
    variance = np.sqrt(security_variance.mul(ticker_weight, axis=0).mul(ticker_weight, axis=1).sum().sum())
    daily_volatility = np.sqrt(1)*variance
    annual_volatility = np.sqrt(252)*variance
    
    portfolio = [expected_returns, daily_volatility, annual_volatility] 
    portfolio.extend(ticker_weight)
    portfolios.append(portfolio)
portfolios = pd.DataFrame(portfolios, columns=['Returns', 'DailyVolatility', 'AnnualVolatility']+[f'Weight_{ticker_name}' for ticker_name in ticker_names])
my_portfolio = portfolios.iloc[-1]
minimum_volatility_portfolio = portfolios.iloc[portfolios['AnnualVolatility'].idxmin()]

plt.figure(figsize=(25,5))
plt.scatter(x=portfolios['AnnualVolatility'], y=portfolios['Returns'], marker='o', s=10, alpha=0.3)
plt.scatter(x=my_portfolio[2], y=my_portfolio[0], color='r', marker='x', s=500)
plt.scatter(x=minimum_volatility_portfolio[2], y=minimum_volatility_portfolio[0], color='r', marker='*', s=500)
plt.grid(True)
plt.show()

display(minimum_volatility_portfolio)
portfolios
```

### Sharpe Ratio
```python
```


### Capital Market Line(CML)
- https://en.wikipedia.org/wiki/Capital_market_line
```python
```

### Capital allocation line(CAL)
- https://en.wikipedia.org/wiki/Capital_allocation_line
```python
```

### Capital Asset Pricing Model(CAPM)
- https://en.wikipedia.org/wiki/Capital_asset_pricing_model
```python
```

### Security Market Line(SML)
- https://en.wikipedia.org/wiki/Security_market_line
```python
```


### Beta
- https://en.wikipedia.org/wiki/Beta_(finance)
```python
```


### Dividend Discount Model(DDM)
- https://en.wikipedia.org/wiki/Dividend_discount_model
```python
```


### Arbitrage Pricing Theory(APT)
- https://en.wikipedia.org/wiki/Arbitrage_pricing_theory
```python

```


## Reference
- https://en.wikipedia.org/wiki/Modern_portfolio_theory
- https://en.wikipedia.org/wiki/Power_transform
- https://en.wikipedia.org/wiki/Volatility_(finance)


