# Investment Package
- https://github.com/ailever/investment
- [Guidance: Analysis](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_analysis.md)
- [Guidance: Dataset](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_dataset.md)
- [Guidance: Model](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_model.md)
- [Guidance: Management](https://github.com/ailever/ailever/blob/master/ailever/investment/guidance_management.md)


## Requirements
```bash
$ apt install openjdk-8-jdk
$ pip install monthdelta
$ pip install tabula-py
```

## Monitoring
```python
from ailever.investment import dashboard

dashboard.run()
```

## Finance Dataset
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

```python
from ailever.investment import Loader 

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
from ailever.investment import prllz_loader

datacore = prllz_loader(baskets=['ARE', 'BXP', 'O'])
datacore.ndarray
datacore.pdframe
```

```python
from ailever.investment import Preprocessor

pre = Preprocessor()
'''for ticker processing'''
pre.pct_change(baskets=['ARE','O','BXP'], window=[1,3,5],kind='ticker')
'''for index preprocessed data attachment'''
pre.pct_change(baskets=['^VIX'], kind='index_full') '''including index ohlcv'''
pre.pct_change(baskets=['^VIX'], kind='index_single') '''Only preprocessed index data

pre.overnight(baskets=['ARE','O','BXP'], kind='index_full') '''including index ohlcv
pre.rolling(baskets=['ARE','O','BXP'], kind='index_full') '''including index ohlcv

pre.date_featuring()
pre.na_handler()

pre.preprocess_list
pre.to_csv(option='dropna')
pre.reset()


```

```python
from ailever.investment import Screener
@staticmethod
Screener.momentum_screener(baskets=['ARE', 'O', 'BXP'], period=10)
Screener.fundamentals_screener(baskets=['ARE', 'O', 'BXP'], sort_by='Marketcap')
Screener.pct_change_screener(baskets=['ARE', 'O', 'BXP'], sort_by=1
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

specifications = dict()
specifications['ARE'] = {'id':1, 'overwritten':True, 'architecture':'lstm00', 'framework':'torch', 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'window':[5,10,20], 'base_columns':['date', 'close', 'volume'], 'packet_size':70, 'prediction_interval':30, 'start':'20180101', 'end':'20210816', 'rep':'ailever', 'message':'message', 'country':'united_states'}
specifications['BXP'] = {'id':2, 'overwritten':True, 'architecture':'lstm00', 'framework':'torch', 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'window':[5,10,20], 'base_columns':['date', 'close', 'volume'], 'packet_size':70, 'prediction_interval':30, 'start':'20180101', 'end':'20210816', 'rep':'ailever', 'message':'message', 'country':'united_states'}
specifications['O'] = {'id':3, 'overwritten':True, 'architecture':'lstm00', 'framework':'torch', 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'window':[5,10,20], 'base_columns':['date', 'close', 'volume'], 'packet_size':70, 'prediction_interval':30, 'start':'20180101', 'end':'20210816', 'rep':'ailever', 'message':'message', 'country':'united_states'}

forecaster = Forecaster()
forecaster.train_trigger(baskets=specifications.keys(), train_specifications=specifications)
forecaster.prediction_trigger(baskets=specifications.keys(), prediction_specifications=specifications)

#forecaster.forecasting_model_registry('remove')
#forecaster.forecasting_model_registry('clearall')
#forecaster.forecasting_model_registry('listdir')
forecaster.forecasting_model_registry('listfiles')
forecaster.model_prediction_result('listfiles')
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
