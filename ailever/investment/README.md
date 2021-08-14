# Investment Package
- https://github.com/ailever/investment


## Monitoring
```python
from ailever.investment import dashboard

dashboard.run()
```

## Finance Dataset
```python
from ailever.investemnet import sectors

tickers = sectors.us_reit()
tickers.list
tickers.pdframe
tickers.subsector

tickers = sectors.us_reit(subsector='Office')
tickers.list
tickers.pdframe
```

```python
from ailever.investment import Loader 

loader = Loader()
dataset = loader.ohlcv_loader(baskets=['ARE', 'O', 'BXP'])
dataset.dict
dataset.log

modules = loader.fmf '--> modules search for fundmentals'
modules = loader.fundamentals_modules_fromyahooquery

dataset = loader.fundamentals_loader(baskets=['ARE', 'O', 'BXP'], modules=list(modules))
dataset.dict
dataset.log

```

```python
from ailever.investment import Preprocessor

pre = Preprocessor()
'''for ticker processing'''
pre.pct_change(baskets=['ARE','O','BXP'], window=[1,3,5],kind='ticker')
'''for index preprocessed data attachment'''
pre.pct_change(baskets=['^VIX'], kind='index_full') '''including index ohlcv'''
pre.pct_change(baskets=['^VIX'], kind='index_single') '''Only preprocessed index data

pre.overnight(baskets=['ARE','O','BXP'], kind='index_full') '''including index ohlcv'''

'''

pre.preprocess_list
pre.to_csv
pre.reset

'''Currently not supporting'''
pre.overnight(baskets=['ARE','O','BXP'], kind='index_full') '''including index ohlcv'''
pre.rolling(baskets=['ARE','O','BXP'], kind='index_full') '''including index ohlcv'''
pre.relative(baskets=['ARE','O','BXP'], kind='index_full') '''including index ohlcv'''
pre.stochastic(baskets=['ARE','O','BXP'], kind='index_full') '''including index ohlcv'''


```

```python
from ailever.investment import Screener
@staticmethod
Screener.momentum_screener(baskets=['ARE', 'O', 'BXP'], period=10)
Screener.fundamentals_screener(baskets=['ARE', 'O', 'BXP'], moduels=moduels, sort_by=sort_by) '''currently not supporting'''
```

```python
from ailever.investment import portfolio_optimizer

portfolio_optimizer(['AMH', 'PSTL', 'SRG'])
```

```python
from ailever.investment import sharp_ratio

```

```python
from ailever.investment import featuring

featuring.finance_base.profitability_ratios()
featuring.finance_base.liquidity_ratios()
featuring.finance_base.activity_ratios()
featuring.finance_base.debt_ratios()
featuring.finance_base.market_ratios()
featuring.finance_base.capital_budgeting_ratios()
featuring.technique_base.rolling()
featuring.technique_base.smoothing()
featuring.technique_base.filtering()
featuring.technique_base.denosing()
featuring.technique.modeling()
featuring.economics_base._()
featuring.extraction(store=False)
```

## Prediction Model

```python
from ailever.investment import Forecaster

train_specifications = dict()
train_specifications['ARE'] = {'architecture':'lstm00', 'framework':'torch', 'loading_process':2, 'storing_process':14, 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'base_columns':['date', 'close', 'volume'], 'packet_size':365, 'prediction_interval':100, 'start':'2015-01-01', 'end':'2017-01-01', 'rep':'ailever', 'message':'message', 'country':'united_states'}
train_specifications['BXP'] = {'architecture':'lstm00', 'framework':'torch',  'loading_process':2, 'storing_process':14, 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'base_columns':['date', 'close', 'volume'], 'packet_size':365, 'prediction_interval':100, 'start':'2015-01-01', 'end':'2017-01-01', 'rep':'ailever', 'message':'message', 'country':'united_states'}
train_specifications['O'] = {'architecture':'lstm00', 'framework':'torch',  'loading_process':2, 'storing_process':14, 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'base_columns':['date', 'close', 'volume'], 'packet_size':365, 'prediction_interval':100, 'start':'2015-01-01', 'end':'2017-01-01', 'rep':'ailever', 'message':'message', 'country':'united_states'}
train_specifications['SPG'] = {'architecture':'lstm00', 'framework':'torch',  'loading_process':2, 'storing_process':14, 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'base_columns':['date', 'close', 'volume'], 'packet_size':365, 'prediction_interval':100, 'start':'2015-01-01', 'end':'2017-01-01', 'rep':'ailever', 'message':'message', 'country':'united_states'}
train_specifications['MPW'] = {'architecture':'lstm00', 'framework':'torch',  'loading_process':2, 'storing_process':14, 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'base_columns':['date', 'close', 'volume'], 'packet_size':365, 'prediction_interval':100, 'start':'2015-01-01', 'end':'2017-01-01', 'rep':'ailever', 'message':'message', 'country':'united_states'}
train_specifications['ADC'] = {'architecture':'lstm00', 'framework':'torch',  'loading_process':2, 'storing_process':14, 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'base_columns':['date', 'close', 'volume'], 'packet_size':365, 'prediction_interval':100, 'start':'2015-01-01', 'end':'2017-01-01', 'rep':'ailever', 'message':'message', 'country':'united_states'}
train_specifications['VTR'] = {'architecture':'lstm00', 'framework':'torch',  'loading_process':2, 'storing_process':14, 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'base_columns':['date', 'close', 'volume'], 'packet_size':365, 'prediction_interval':100, 'start':'2015-01-01', 'end':'2017-01-01', 'rep':'ailever', 'message':'message', 'country':'united_states'}

forecaster = Forecaster()
forecaster.train_trigger(baskets=train_specifications.keys(), train_specifications=train_specifications)
forecaster.model_registry('listfiles')
```


```python
from ailever.investment import Forecaster

train_specifications = dict()
train_specifications['ARE'] = {'architecture':'lstm00', 'loading_process':2, 'storing_process':14, 'device':'cpu', 'batch_size':10, 'shuffle':False, 'drop_last':False, 'epochs':100, 'base_columns':['date', 'close', 'volume'], 'packet_size':365, 'prediction_interval':100, 'start':'2015-01-01', 'end':'2017-01-01', 'rep':'ailever', 'message':'message', 'country':'united_states'}

model = Forecaster()
model.train_trigger(baskets=['ARE'], train_specifications=train_specifications)
model.evaludation_trigger()
model.upload()

model.monitoring.train_period
model.monitoring.backtesting_period
model.monitoring.packet_info

model.max_profit()
model.summary()
```


```python
from ailever.investment import Forecasters

model = Forecaster(onasset='reits', target='transaction time', id=1)
model.train_trigger()
model.evaludation_trigger()
model.prediction()
model.upload()

model.monitoring.train_period
model.monitoring.backtesting_period
model.monitoring.packet_info

model.max_profit()
model.summary()
```

## Management Model

```python
from ailever.investment import strategy_models

model = strategy_models(id=1)

model.monitoring.train_period
model.monitoring.backtesting_period
model.monitoring.packet_info

model.expected_profit()
model.summary()
```

## Misc

```python
from ailever.investment import Logger

logger = Logger() -> See below for DEFAUlT-CONFIG
logger.normal_logger.info("message")
logger.normal_logger.warnings("message")
logger.normal_logger.error("message")
logger.normal_logger.exception("message") <-- Works only in Exception loop

logger.dev_logger.debug("message")
logger.dev_logger.info("message")

DEFAULT CONFIG:

config = {
            "version": 1,
            "formatters": {
                "simple": {"format": "[%(name)s] %(message)s"},
                "complex":{
                    "format": "[%(asctime)s]/[%(name)s]/[%(filename)s:%(lineno)d]/[%(levelname)s]/[%(message)s]"},
                },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "level": "DEBUG",
                    },
                "file": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(log_dirname, "meta.log"),
                    "formatter": "complex",
                    "level": "INFO",
                    },
                },
            "root": {"handlers": ["console", "file"], "level": "WARNING"},
            "loggers": {"normal": {"level": "INFO"}, "dev": {"level": "DEBUG"},},
            }
```
