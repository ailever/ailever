# Analysis Package
```python
from ailever.analysis import dashboard
dashboard.run()
```

```python
from ailever.analysis import EDA
from ailever.dataset import UCI

frame = UCI.breast_cancer(download=False)
eda = EDA(frame)
eda.table_definition()
eda.cleaning(as_float=None, as_int=None, as_date=None)
eda.attributes_specification()
eda.univariate_frequency()
eda.univariate_percentile()
eda.univariate_conditional_frequency()
eda.univariate_conditional_percentile()
eda.multivariate_frequency(base_column='Class', column_sequence=['breast-quad', 'irradiat'])
eda.information_value(target_column='target', target_event=1, view='result')
eda.feature_importance()
```

## Numerical Variable Analysis
### Percentile-based
```python
from ailever.analysis import EDA
from ailever.dataset import UCI

frame = UCI.adult(download=False)
eda = EDA(frame)
eda.cleaning(as_int=['age'], as_float=['capital-gain', 'education-num'])
eda.univariate_percentile(view='summary')
eda.univariate_conditional_percentile(base_column='age', view='summary').sort_values('CohenMeasureRank')
```


## Categorical Variable Analysis
```python
from ailever.analysis import EDA
from ailever.dataset import UCI

#frame = UCI.breast_cancer(download=False)
frame = UCI.adult(download=False)
eda = EDA(frame)
eda.cleaning(as_int=['age'], as_float=['capital-gain', 'education-num'])
eda.univariate_frequency(view='summary').sort_values('Rank')
eda.univariate_conditional_frequency(base_column='age')['native-country']
eda.multivariate_frequency(base_column='age', column_sequence=['native-country'])
```
