# investment

```python
from ailever.investment import parallelize

prllz = parallelize(path='reits_csvs/', object_foramt='csv', base_column='close', date_column='date', period=100)
prllz.ndarray
prllz.pdframe
```

```python
from ailever.investment import reits_screening

reits_screening(path='reits_csvs/', period=100)
```
