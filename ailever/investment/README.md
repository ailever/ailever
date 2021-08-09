# investment
- https://github.com/ailever/investment


```python
from ailever.investment import dashboard

dashboard.run()
```

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
```

```python
from ailever.investment import screener

screened = screener(baskets=['ARE', 'O', 'BXP'], period=10)
screened.ndaary
screened.pdframe
screened.list

```

```python
from ailever.investment import reits_screening

reits_screening(path='financedatasets', period=100)
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


```python

from ailever.investment import processor
from ailever.investment import integrated_loader

df = integrated_loader(baskets=['ARE'], path='financedatasets')
p = Processor(df['ARE'])

p._overnight() -> return self
p._overnight().result -> return dataframe
p._overnight(output="pdframe").overnight
p._overnight(output="ndarray").overnight

p._rolling(column='close', window=10) -> return self
p._rolling().result -> return dataframe
p._rolling(output="pdframe").rolling
p._rolling(output="ndarray").rolling

```


```python
from ailever.investment import Forecasters

model = Forecaster(onasset='reits', target='price', id=1)
model.train_trigger()
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


```python
from ailever.investment import strategy_models

model = strategy_models(id=1)

model.monitoring.train_period
model.monitoring.backtesting_period
model.monitoring.packet_info

model.expected_profit()
model.summary()
```

#Misc

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
