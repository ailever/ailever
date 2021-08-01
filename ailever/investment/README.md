# investment
- https://github.com/ailever/investment


```python
from ailever.investment import initialize

initialize()
```

```python
from ailever.investment import integrated_loader

integrated_loader(baskets=['ARE', 'BIX'])
```

```python
from ailever.investment import parallelize

prllz_objs = parallelize(path='.financedatasets/', object_foramt='csv', base_column='close', date_column='date', period=100)
prllz_objs.ndarray
prllz_objs.pdframe
```


```python
from ailever.investment import reits_screening

reits_screening(path='.financedatasets/', period=100)
```


```python
from ailever.investment import portfolio_optimizer

portfolio_optimizer(['AMH', 'PSTL', 'SRG'])
```


```python
from ailever.investment import sharp_ratio

```



