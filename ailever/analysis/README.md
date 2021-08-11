# Analysis Package
```python
from ailever.analysis import dashboard
dashboard.run()
```

```python
from ailever.analysis import EDA

eda = EDA(frame)
eda.table_definition()
eda.attributes_specification()
eda.univariate_frequency()
eda.univariate_percentile()
eda.univariate_conditional_frequency()
eda.univariate_conditional_percentile()
eda.multivariate_frequency(base_column='Class', column_sequence=['breast-quad', 'irradiat'])
eda.information_value(target_column='target', target_event=1, view='result')
eda.feature_importance()
```
