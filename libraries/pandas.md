## [Data Analysis] | [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/index.html) | [github](https://github.com/pandas-dev/pandas)


### Resampling
```python

```
```python
import FinanceDataReader as fdr

stock = fdr.DataReader('005930', start='2020-04-01')['Close']
stock.resample('D').mean().interpolate('linear')
```
