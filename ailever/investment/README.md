# investment
- https://github.com/ailever/investment


```python
from ailever.investment import dashboard

dashboard.run()
```

```python
from ailever.investment import initialize

# StockListing Loader
initialize()
```

```python
from ailever.investment import integrated_dataloader

datacore = integrated_dataloader(baskets=['ARE', 'O', 'BXP'], path='us_reits', on_asset='reits')
datacore.dict
datacore.log

datacore = integrated_loader(baskets=['ARE', 'O', 'BXP'], path='us_reits', on_asset='reits')
datacore.dict
datacore.log
```

```python
from ailever.investment import parallelize

prllz_objs = parallelize(path='financedatasets', object_format='csv', base_column='close', date_column='date', period=100)
prllz_objs.ndarray
prllz_objs.pdframe
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
from ailever.investment import forecasting_models

model = forecasting_models.reits(id=1)
model.train()
model.prediction()
model.upload()

model.monitoring.train_period
model.monitoring.backtesting_period
model.monitoring.packet_info

model.max_profit()
model.summary()
```


```python
from ailever.investment import capturing_models

model = capturing_models.reits(id=1)
model.train()
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

model = strategy_models.reits(id=1)

model.monitoring.train_period
model.monitoring.backtesting_period
model.monitoring.packet_info

model.expected_profit()
model.summary()
```







