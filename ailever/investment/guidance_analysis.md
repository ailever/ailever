# Financial Investment Analysis

## Market Information
```python
from ailever.investment import market_information
import FinanceDataReader as fdr

df1 = market_information()
df2 = fdr.StockListing('KRX-MARCAP')
pd.merge(df1, df2, how='left', left_on='Symbol', right_on='Code').drop(['Name_y', 'Market_y'], axis=1).rename(columns={'Market_x':'Market', 'Name_x':'Name'})
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
