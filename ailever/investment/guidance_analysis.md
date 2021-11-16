- [Guidance: Analysis](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_analysis.md)
- [Guidance: Dataset](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_dataset.md)
- [Guidance: Model](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_model.md)
- [Guidance: Management](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_management.md)

---

# Financial Investment Analysis

## Market Information
```python
from ailever.investment import market_information

market_information(market_cap=True)
```

## Interest Rate
```python
from ailever.analysis.finance import earnings
earnings(principal=1000, periods=30, max_rate=20, standard='daily')
```
```python
from ailever.analysis.finance import earnings
earnings(principal=1000, periods=30, max_rate=20, standard='daily')
from ailever.analysis import DataTransformer
import FinanceDataReader as fdr

start = '2010-01-01'
frame = fdr.DataReader('005390')
frame['Close'].loc[start:].plot(figsize=(25,7))

frame = DataTransformer.rel_diff(frame, target_columns=['Close'], only_transform=True, keep=False, binary=False, periods=[2,3,4,5,6,7,8,9,10,15,20,60,100,200], within_order=1)
frame.loc[start:].plot(figsize=(25,7))

frame = DataTransformer.ew_binning(frame, target_columns=frame.columns.to_list(), bins=[2,5], only_transform=True, keep=False)
frame.loc[start:].plot(figsize=(25,7))
```


## Correlation
```python
from ailever.forecast import TSA
from ailever.investment import prllz_loader
from ailever.investment import sectors
from ailever.investment import Screener

tickers = sectors.us_reit()
screened_tickers = Screener.fundamentals_screener(baskets=tickers.list, sort_by='Marketcap')
datacore = prllz_loader(baskets=screened_tickers[:10], period=100)

tsa = TSA(datacore.pdframe, screened_tickers[0])
tsa.Correlation(datacore.pdframe, column_sequence=screened_tickers[:5])
```

## Decomposition
```python
from ailever.forecast import TSA
from ailever.investment import prllz_loader
from ailever.investment import sectors
from ailever.investment import Screener

tickers = sectors.us_reit()
screened_tickers = Screener.fundamentals_screener(baskets=tickers.list, sort_by='Marketcap')
datacore = prllz_loader(baskets=screened_tickers[:10], period=100)

tsa = TSA(datacore.pdframe, screened_tickers[0])
tsa.STL()
```

## Security Returns Analysis
```python
from ailever.analysis import EDA
from ailever.investment import prllz_loader
from ailever.investment import sectors
from ailever.investment import Screener

tickers = sectors.us_reit()
screened_tickers = Screener.fundamentals_screener(baskets=tickers.list, sort_by='Marketcap')
datacore = prllz_loader(baskets=screened_tickers[:10], period=100)

eda = EDA(datacore.pdframe.pct_change().fillna(0), verbose=False)
eda.univariate_percentile(percent=5, view='summary', visual_on=True)
```

## Security Information Values
```python

```
