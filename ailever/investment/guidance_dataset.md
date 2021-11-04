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

