- [Guidance: Analysis](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_analysis.md)
- [Guidance: Dataset](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_dataset.md)
- [Guidance: Model](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_model.md)
- [Guidance: Management](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_management.md)

---

## Framework supporting financial dataset
### FianaceDataReader
`fundamentals`
```python
import FinanceDataReader as fdr
fdr.StockListing('KRX-MARCAP')
```
`market indicies`
- **FUTURE**: 'NG', 'GC', 'SI', 'HG', 'CL'
- **MARKET**: 'KS11', 'KQ11', 'KS50', 'KS100', 'KRX100', 'KS200', 'DJI', 'IXIC', 'US500', 'RUTNU', 'VIX', 'JP225', 'STOXX50', 'HK50', 'CSI300', 'TWII', 'HNX30', 'SSEC', 'UK100', 'DE30', 'FCHI'
- **EXCHANGE RATE**: 'USD/KRW', 'USD/EUR', 'USD/JPY', 'CNY/KRW', 'EUR/USD', 'USD/JPY', 'JPY/KRW', 'AUD/USD', 'EUR/JPY', 'USD/RUB'
- **GOVERNMENT BOND**: 'KR1YT=RR', 'KR2YT=RR', 'KR3YT=RR', 'KR4YT=RR', 'KR5YT=RR', 'KR10YT=RR', 'KR20YT=RR', 'KR30YT=RR', 'KR50YT=RR', 'US1MT=X', 'US3MT=X', 'US6MT=X', 'US1YT=X', 'US2YT=X', 'US3YT=X', 'US5YT=X', 'US7YT=X','US10YT=X', 'US30YT=X'
- **CRYPTOCURRENCY**: 'BTC/KRW','ETH/KRW','XRP/KRW','BCH/KRW','EOS/KRW','LTC/KRW','XLM/KRW', 'BTC/USD','ETH/USD','XRP/USD','BCH/USD','EOS/USD','LTC/USD','XLM/USD'

```python
import FinanceDataReader as fdr
fdr.DataReader('KS11')
```
`tickers`
```python
import FinanceDataReader as fdr
fdr.DataReader('005930')
```

### Yahooquery
`fundamentals`
```python
from yahooquery import Ticker
import pandas as pd

ticker = Ticker('ARE')
pd.DataFrame(ticker.summary_detail).T
```
`market indicies`
```python
```
`tickers`
```python
from yahooquery import Ticker
ticker = Ticker('ARE')
ticker.history()
```


### Crawl: Finviz
```python
import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get('https://finviz.com/quote.ashx?t=AAPL', headers=headers)
soup = BeautifulSoup(response.text)
target_table_tag = soup.find('table', attrs={'class': 'snapshot-table2'}) #tables = soup.find_all('table')
df = pd.read_html(str(target_table_tag))[0]

df.columns = ['key', 'value'] * 6
df_list = [df.iloc[:, i*2: i*2+2] for i in range(6)]
df_factor = pd.concat(df_list, ignore_index=True)
df_factor.set_index('key', inplace=True)
df_factor
```



## Initialization
```python
import pandas as pd
from ailever.investment import market_information, Loader

loader = Loader()
loader.into_local()

df = market_information(market_cap=False)
Df = loader.from_local(baskets=df[df.Market=='KOSPI'].Symbol.to_list(), mode='Close')
pd.DataFrame(data=Df[0], columns=Df[1].Name.to_list())
```


## Market Information
```python
from ailever.investment import market_information
df = market_information(baskets=None, only_symbol=False, inverse_mapping=False, market_cap=False)
df[0]
```

```python
from ailever.investment import market_information
df = market_information(baskets=['삼성전자', 'SK하이닉스'], only_symbol=True, inverse_mapping=False, market_cap=False)
df
```

```python
from ailever.investment import market_information
df = market_information(baskets=['005930', '000660'], only_symbol=False, inverse_mapping=True, market_cap=False)
df
```

## Sector
```python
from ailever.investment import sectors

tickers = sectors.us_reit()
tickers.list
tickers.pdframe
tickers.subsector

tickers = sectors.us_reit(subsector='Office')
tickers.list
tickers.pdframe
```

## Data Vendor
```python
```

## Parallelizer
```python
from ailever.investment import prllz_loader

datacore = prllz_loader(baskets=['ARE', 'BXP', 'O'])
datacore.ndarray
datacore.pdframe
```

```python
from ailever.investment import market_information
from ailever.investment import parallelize
import pandas as pd

df = market_information()
baskets = df.loc[lambda x: x.Market == 'KOSPI'].dropna().reset_index().drop('index', axis=1).Symbol.to_list()
sample_columns = pd.read_csv('.fmlops/feature_store/1d/005390.csv').columns.to_list()

DTC = parallelize(baskets=baskets, path='.fmlops/feature_store/1d', base_column='Close', date_column='Date', columns=sample_columns)
DTC.pdframe
```

## Integrated Loader
```python
from ailever.investment import market_information
from ailever.investment import Loader 

df = market_information().dropna()
df[df.Industry.str.contains('리츠')].loc[lambda x:x.Market == 'NYSE']

loader = Loader()
dataset = loader.ohlcv_loader(baskets=['ARE', 'O', 'BXP'])
dataset.dict
dataset.log

modules = loader.fmf  # '--> modules search for fundmentals'
modules = loader.fundamentals_modules_fromyahooquery

dataset = loader.fundamentals_loader(baskets=['ARE', 'O', 'BXP'], sort_by='Marketcap')
dataset.dict
dataset.log
```


```python
from ailever.investment import Loader
loader = Loader()
loader.into_local()
```

## Preprocessor
```python
from ailever.investment import Preprocessor

pre = Preprocessor() #'''for ticker processing'''
pre.pct_change(baskets=['ARE','O','BXP'], window=[1,3,5],kind='ticker') #'''for index preprocessed data attachment'''
pre.pct_change(baskets=['^VIX'], kind='index_full') #'''including index ohlcv'''
pre.pct_change(baskets=['^VIX'], kind='index_single') #'''Only preprocessed index data

pre.overnight(baskets=['ARE','O','BXP'], kind='index_full') #'''including index ohlcv
pre.rolling(baskets=['ARE','O','BXP'], kind='index_full') #'''including index ohlcv

pre.date_featuring()
pre.na_handler()

pre.preprocess_list
pre.to_csv(option='dropna')
pre.reset()
```

