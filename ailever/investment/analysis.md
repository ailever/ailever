# Financial Investment Analysis

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
