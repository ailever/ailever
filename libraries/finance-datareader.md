## [Stock Market] | [finance-datareader](https://github.com/FinanceData/FinanceDataReader/wiki/Users-Guide) | [github](https://github.com/FinanceData/FinanceDataReader)

```python
import matplotlib.pyplot as plt
import numpy as np
import FinanceDataReader as fdr
import pandas as pd
import seaborn as sns

stock_list = fdr.StockListing('KRX')
stock_list = stock_list.set_index('Name')
symbol = stock_list[stock_list.index == '삼성전자'].Symbol.values.squeeze()
_, axes = plt.subplots(6,1, figsize=(13,20))

stock = fdr.DataReader(str(symbol), start='2020-01-01')['Close']
stock.plot(ax=axes[0])
stock.diff().plot(ax=axes[1])
sns.distplot(stock.diff(), kde=True, ax=axes[2])
pd.plotting.lag_plot(stock.diff(), lag=5, ax=axes[3])
pd.plotting.autocorrelation_plot(stock.diff().dropna(), ax=axes[4])
stock.cumsum().plot(ax=axes[5])

axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
axes[3].grid(True)
axes[4].grid(True)
axes[5].grid(True)
```
