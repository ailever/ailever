# Analysis Package
`dashboard`
```python
from ailever.analysis import dashboard
dashboard.run()
```

`options`
```python
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
plt.rcParams["font.family"] = 'NanumBarunGothic'
```

## Exploratory Data Analysis
```python
from ailever.dataset import UCI
from ailever.analysis import EDA

frame = UCI.breast_cancer(download=False)
eda = EDA(frame, verbose=False)
eda.table_definition()
eda.cleaning(as_float=None, as_int=None, as_str=None, as_category=None, as_date=None, verbose=False)
eda.attributes_specification()

eda.univariate_frequency()
eda.univariate_conditional_frequency()
eda.frequency_visualization(base_column='Class', column_sequence=['breast-quad', 'irradiat'])

eda.univariate_percentile()
eda.univariate_conditional_percentile()

eda.information_value(target_column='target', target_event=1, view='result')
eda.feature_importance()
```

### Table Attributes Definition
```python
from ailever.dataset import UCI
from ailever.analysis import EDA

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=True)
eda.cleaning(as_float=None, as_int=None, as_date=None, as_str=None, as_category=None, verbose=True)
# column classification by column-type after cleaning
eda.null_columns     # frame.columns = eda.null_columns + eda.not_null_columns
eda.not_null_columns # eda.not_null_columns = eda.integer_columns + eda.float_columns + eda.string_columns + eda.category_columns
eda.integer_columns
eda.float_columns
eda.string_columns
eda.category_columns

eda.table_definition()
eda.attributes_specification(visual_on=True)
eda.results['attributes_specification']['MVRate'].value_counts().sort_index()
# column classification by column-type after specifying
# eda.not_null_columns = eda.normal_columns + eda.abnormal_columns = eda.numeric_columns + eda.categorical_columns
eda.normal_columns    # columns not having missing values 
eda.abnormal_columns  # columns having missing values
eda.numeric_columns
eda.categorical_columns
```

### Exploratory Numerical Variable Analysis
#### Percentile-based
```python
from ailever.dataset import UCI
from ailever.analysis import EDA

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)
eda.cleaning(as_int=['age'], as_float=['capital-gain', 'education-num'], as_str=all, verbose=False)

#eda.frame.describe().T
#eda.univariate_percentile(percent=5, view='summary', visual_on=True)
eda.univariate_conditional_percentile(base_column='age', percent=5, view='summary', visual_on=False).loc[lambda x: x.CohenMeasureRank <= 10].sort_values('CohenMeasureRank', ascending=True)
```

#### Correlation Analysis
```python

```




### Exploratory Categorical Variable Analysis
#### Frequency-based
```python
from ailever.dataset import UCI
from ailever.analysis import EDA

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)
eda.cleaning(as_int=['age'], as_float=['capital-gain', 'education-num'], as_category=all, verbose=False)

#eda.frame.describe(include='category').T
#eda.univariate_frequency(view='summary').loc[lambda x: x.Rank <= 1]

BASE = 'age'
SEQ = ['native-country']
eda.frequency_visualization(base_column=BASE, column_sequence=SEQ)
eda.univariate_conditional_frequency(base_column=BASE)[SEQ[-1]]
```


#### Information Value
```python
from ailever.dataset import UCI
from ailever.analysis import EDA

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)

eda.cleaning(as_int=['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'education-num', 'hours-per-week'], as_str=all, verbose=False)
eda.information_values(target_column='age')
eda.iv_summary['column'] # eda.iv_summary['result']

#%%
eda.iv_summary['instance'].loc[lambda x: x.InstanceIVRank <= 3]
eda.iv_summary['instance'].sort_values('InstanceIVRank').loc[lambda x: x.Column == 'capital-gain'].loc[lambda x: x.InstanceIVRank <= 10]
eda.iv_summary['instance'].sort_values('Column').loc[lambda x: x.InstanceIVRank <= 3]
```

#### Feature Importance
```python
from ailever.dataset import UCI
from ailever.analysis import EDA

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)
eda.cleaning(as_int=['age', 'fnlwgt', 'education-num', 'hours-per-week'], as_float=['capital-gain', 'capital-loss'])
#eda.attributes_specification(visual_on=False)

eda.feature_importance(target_column='marital-status', target_instance_covering=5, decimal=1)

#%%
eda.fi_summary['fitting_table']
eda.fi_summary['feature_importance']
eda.fi_summary['decision_tree']
```

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


#### Permutation Importance
```python
import eli5 
from eli5.sklearn import PermutationImportance 
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier().fit(X_train, y_train)

perm = PermutationImportance(model, scoring = "accuracy", random_state = 22).fit(X_val, y_val) 
eli5.show_weights(perm, top = 20, feature_names = X_val.columns.tolist())
```


## Data-Preprocessing : DataTransformer
`time_splitor`
```python
from ailever.analysis import DataTransformer
from ailever.dataset import SMAPI

frame = SMAPI.co2(download=False)
frame['date'] = frame.index
DataTransformer.time_splitor(frame)
```
```python
from ailever.analysis import EDA
from ailever.analysis import DataTransformer
from ailever.dataset import SMAPI

frame = SMAPI.co2(download=False)
frame['date'] = frame.index
frame = DataTransformer.time_splitor(frame)
frame['target'] = frame['co2'].diff().fillna(0).apply(lambda x: 1 if x>0 else 0)

eda = EDA(frame, verbose=False)
eda.information_values(target_column='target', target_event=1, verbose=False)
eda.iv_summary['column']
```

`ew_binning`
```python
from ailever.analysis import DataTransformer
from ailever.dataset import UCI

frame = UCI.adult(download=False)
#DataTransformer.empty()
#DataTransformer.build()
DataTransformer.ew_binning(frame, target_columns=['capital-gain', 'capital-loss', 'hours-per-week'], bins=[4, 100], only_transform=True, keep=False)
```
```python
from ailever.dataset import SMAPI
from ailever.analysis import EDA
from ailever.analysis import DataTransformer
#plt.rcParams["font.family"] = 'NanumBarunGothic'

frame = SMAPI.co2(download=False)
frame = DataTransformer.ew_binning(frame, target_columns=['co2'], bins=[4, 10, 20], only_transform=False, keep=False)
frame['target'] = frame['co2'].diff().fillna(0).apply(lambda x: 1 if x>0 else 0)

eda = EDA(frame, verbose=False)
eda.information_values(target_column='target', target_event=1)
```
```python
from ailever.analysis import DataTransformer
from ailever.analysis import EDA
from ailever.dataset import SMAPI


frame = SMAPI.co2(download=False)
frame = DataTransformer.ew_binning(frame, target_columns=['co2'], bins=[4, 10, 20], only_transform=False, keep=False)
frame = DataTransformer.derivatives(frame, target_columns=['co2'], only_transform=False, keep=False)

eda = EDA(frame, verbose=False)
eda.information_values(target_column='co2_increasing_1st', target_event=1)
```
```python
from ailever.dataset import SMAPI
from ailever.analysis import EDA
from ailever.analysis import DataTransformer
#import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = 'NanumBarunGothic'

frame = SMAPI.co2(download=False).dropna()
frame = DataTransformer.ew_binning(frame, target_columns=['co2'], bins=[4, 10, 20], only_transform=False, keep=False)
frame['target'] = frame['co2'].diff().fillna(0).apply(lambda x: 1 if x>0 else 0)

eda = EDA(frame, verbose=False)
eda.cleaning(as_float=['co2', 'co2_ew4bins', 'co2_ew10bins', 'co2_ew20bins'], as_int=['target'])
eda.feature_importance(target_column='target', target_instance_covering=2, decimal=1)
```


`ef_binning`
```python
from ailever.dataset import UCI
from ailever.analysis import DataTransformer

frame = UCI.adult(download=False)
#DataTransformer.empty()
#DataTransformer.build()
DataTransformer.ef_binning(frame, target_columns=['age', 'fnlwgt'], bins=[4], only_transform=True, keep=False) # Check DataTransformer.storage_box[-1] when keep == True
```
```python
from ailever.dataset import SMAPI
from ailever.analysis import DataTransformer
from ailever.analysis import EDA

frame = SMAPI.co2(download=False)
frame = DataTransformer.ef_binning(frame, target_columns=['co2'], bins=[4, 10, 20], only_transform=False, keep=False) # Check DataTransformer.storage_box[-1] when keep == True
frame['target'] = frame['co2'].diff().fillna(0).apply(lambda x: 1 if x>0 else 0)

eda = EDA(frame, verbose=False)
eda.information_values(target_column='target', target_event=1)
```
```python
from ailever.dataset import SMAPI
from ailever.analysis import DataTransformer
from ailever.analysis import EDA

frame = SMAPI.co2(download=False)
frame = DataTransformer.ef_binning(frame, target_columns=['co2'], bins=[4, 10, 20], only_transform=False, keep=False) # Check DataTransformer.storage_box[-1] when keep == True
frame = DataTransformer.derivatives(frame, target_columns=['co2'], only_transform=False, keep=False)

eda = EDA(frame, verbose=False)
eda.information_values(target_column='co2_increasing_1st', target_event=1)
```
```python
from ailever.dataset import SMAPI
from ailever.analysis import EDA
from ailever.analysis import DataTransformer
#import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = 'NanumBarunGothic'

frame = SMAPI.co2(download=False).dropna()
frame = DataTransformer.ef_binning(frame, target_columns=['co2'], bins=[4, 10, 20], only_transform=False, keep=False)
frame['target'] = frame['co2'].diff().fillna(0).apply(lambda x: 1 if x>0 else 0)

eda = EDA(frame, verbose=False)
eda.cleaning(as_float=['co2', 'co2_ef4bins', 'co2_ef10bins', 'co2_ef20bins'], as_int=['target'])
eda.feature_importance(target_column='target', target_instance_covering=2, decimal=1)
```

### Data Cleaning

### Data Transformation


## Time Series Analysis
`spliting`
```python
from ailever.dataset import SMAPI
from ailever.analysis import EDA
from ailever.analysis import DataTransformer
#import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = 'NanumBarunGothic'

frame = SMAPI.co2(download=False).dropna().reset_index().rename(columns={'index':'date'})
frame = DataTransformer.time_splitor(frame, date_column='date')
frame['target'] = frame['co2'].diff().fillna(0).apply(lambda x: 1 if x>0 else 0)

eda = EDA(frame, verbose=False)
eda.cleaning(as_float=['co2'], as_int=['TS_year', 'TS_quarter', 'TS_month', 'TS_week', 'TS_day', 'TS_daysinmonth', 'TS_sequence', 'target'])
eda.information_values(target_column='target', target_event=1)
eda.feature_importance(target_column='target', target_instance_covering=2, decimal=1)
```

`binning`
```python
from ailever.dataset import SMAPI
from ailever.analysis import EDA
from ailever.analysis import DataTransformer
#import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = 'NanumBarunGothic'

frame = SMAPI.co2(download=False).dropna().reset_index().rename(columns={'index':'date'})
frame = DataTransformer.ew_binning(frame, target_columns=['co2'], bins=[4, 10, 20], only_transform=False, keep=False)
frame = DataTransformer.ef_binning(frame, target_columns=['co2'], bins=[4, 10, 20], only_transform=False, keep=False)
frame['target'] = frame['co2'].diff().fillna(0).apply(lambda x: 1 if x>0 else 0)

eda = EDA(frame, verbose=False)
eda.cleaning(as_float=['co2', 'co2_ew4bins', 'co2_ew10bins', 'co2_ew20bins', 'co2_ef4bins', 'co2_ef10bins', 'co2_ef20bins'], as_int=['target'])
eda.information_values(target_column='target', target_event=1)
eda.feature_importance(target_column='target', target_instance_covering=2, decimal=1)
```

`smoothing`
```python
from ailever.dataset import SMAPI
from ailever.analysis import EDA
from ailever.analysis import DataTransformer
#import matplotlib.pyplot as plt
#plt.rcParams["font.family"] = 'NanumBarunGothic'

frame = SMAPI.co2(download=False).dropna().reset_index().rename(columns={'index':'date'})
frame = DataTransformer.sequence_smoothing(frame, target_column='co2', date_column='date', freq='D', smoothing_order=1, decimal=1, including_model_object=False, only_transform=False, keep=True)
frame['target'] = frame['co2'].diff().fillna(0).apply(lambda x: 1 if x>0 else 0)

eda = EDA(frame, verbose=False)
eda.cleaning(as_float=['co2', 'co2_smt101X0000', 'co2_smt202X0000', 'co2_smt010X0000', 'co2_smt111X0000', 'co2_smt212X0000', 'co2_smt000X0107', 'co2_smt010X0107', 'co2_smt111X0107', 'co2_smt212X0107'], as_int=['target'])
eda.information_values(target_column='target', target_event=1)
eda.feature_importance(target_column='target', target_instance_covering=10, decimal=1)
```

```python
from ailever.dataset import SMAPI
import statsmodels.tsa.api as smt

frame = SMAPI.co2(download=False).dropna()
frame = frame.asfreq('M').fillna(method='bfill').fillna(method='ffill')

trend = [None, 'c', 't', 'ct']
model = smt.SARIMAX(frame['co2'], order=(1,0,1), seasonal_order=(1,1,2,7), trend=trend[0], freq='M', simple_differencing=False)
model = model.fit(disp=False)

frame['feature_210X1027'] = model.predict()
frame
```
- model.states.filtered
- model.states.filtered_cov
- model.states.predicted
- model.states.predicted_cov
- model.states.smoothed
- model.states.smoothed_cov

- model.summary()
- model.params
- model.arparams
- model.maparams
- model.seasonalarparams
- model.seasonalmaparams
- model.pvalues
- model.tvalues
- model.zvalues
- model.mse
- model.mae

