# Analysis Package
```python
from ailever.analysis import dashboard
dashboard.run()
```

## Exploratory Data Analysis
```python
from ailever.analysis import EDA
from ailever.dataset import UCI

frame = UCI.breast_cancer(download=False)
eda = EDA(frame, verbose=False)
eda.table_definition()
eda.cleaning(as_float=None, as_int=None, as_str=None, as_category=None, as_date=None, verbose=False)
eda.attributes_specification()
eda.univariate_frequency()
eda.univariate_percentile()
eda.univariate_conditional_frequency()
eda.univariate_conditional_percentile()
eda.multivariate_frequency(base_column='Class', column_sequence=['breast-quad', 'irradiat'])
eda.information_value(target_column='target', target_event=1, view='result')
eda.feature_importance()
```

### Table Attributes Definition
```python
from ailever.analysis import EDA
from ailever.dataset import UCI

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=True)
eda.table_definition()
eda.attributes_specification()
eda.cleaning(as_float=None, as_int=None, as_date=None, as_str=None, as_category=None, verbose=True)
```

### Exploratory Numerical Variable Analysis
#### Percentile-based
```python
from ailever.analysis import EDA
from ailever.dataset import UCI

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)
eda.cleaning(as_int=['age'], as_float=['capital-gain', 'education-num'], as_str=all, verbose=False)

#eda.frame.describe().T
#eda.univariate_percentile(percent=5, view='summary')
eda.univariate_conditional_percentile(base_column='age', percent=5, view='summary').loc[lambda x: x.CohenMeasureRank <= 10]
```

#### Correlation Analysis
```python

```




### Exploratory Categorical Variable Analysis
#### Frequency-based
```python
from ailever.analysis import EDA
from ailever.dataset import UCI

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)
eda.cleaning(as_int=['age'], as_float=['capital-gain', 'education-num'], as_category=all, verbose=False)

#eda.frame.describe(include='category').T
#eda.univariate_frequency(view='summary').loc[lambda x: x.Rank <= 1]

BASE = 'age'
SEQ = ['native-country']
eda.multivariate_frequency(base_column=BASE, column_sequence=SEQ)
eda.univariate_conditional_frequency(base_column=BASE)[SEQ[-1]]
```


#### Information Value
```python
from ailever.analysis import EDA
from ailever.dataset import UCI

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)

eda.cleaning(as_int=['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'education-num', 'hours-per-week'], as_str=all, verbose=True)
eda.information_values(target_column='age')

#eda.iv_summary['result']
eda.iv_summary['column']
eda.iv_summary['instance'].loc[lambda x: x.InstanceIVRank <= 3]
eda.iv_summary['instance'].sort_values('Column').loc[lambda x: x.InstanceIVRank <= 3]
```

#### Feature Importance
`xgboost`
```python
from xgboost import XGBClassifier
from xgboost import plot_importance

model = XGBClassifier(random_state=11)
model.fit(X_train, y_train)

plot_importance(model, max_num_features=20)
```
```python
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

model = XGBClassifier(random_state=11)
model.fit(X_train, y_train)

ft_importance_values = model.feature_importances_
ft_series = pd.Series(ft_importance_values, index = X_train.columns)
ft_top20 = ft_series.sort_values(ascending=False)[:20]

plt.figure(figsize=(8,6))
plt.title('Feature Importance Top 20')
sns.barplot(x=ft_top20, y=ft_top20.index)
plt.show()
```

`decision-tree`
```python
```

#### Permutation Importance
```python
import eli5 
from eli5.sklearn import PermutationImportance 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier().fit(X_train, y_train)

perm = PermutationImportance(model, scoring = "accuracy", random_state = 22).fit(X_val, y_val) 
eli5.show_weights(perm, top = 20, feature_names = X_val.columns.tolist())
```


## Data-Preprocessing
### Data Cleaning

### Data Transformation


