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
from ailever.investment import integrated_loader

integrated_loader(baskets=['ARE', 'BIX'], path='us_reits')
```

```python
from ailever.investment import parallelize

prllz_objs = parallelize(path='us_reits', object_format='csv', base_column='close', date_column='date', period=100)
prllz_objs.ndarray
prllz_objs.pdframe
```


```python
from ailever.investment import reits_screening

reits_screening(path='us_reits', period=100)
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

featuring.extraction(store=False)
```


```python
from ailever.investment import forecasting_model

model = forecasting_model.reits(id=1)
model.train_period
model.backtesting_period
model.packet_info
model.max_profit()
model.summary()
```


```python
from ailever.investment import capturing_model

model = capturing_model.reits(id=1)
model.train_period
model.backtesting_period
model.packet_info
model.max_profit()
model.summary()
```


```python
from ailever.investment import strategy_model

model = strategy_model.reits(id=1)
model.expected_profit()
model.summary()
```







