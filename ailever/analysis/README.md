# Analysis Package
- https://github.com/ailever/analysis
- [Dataset](https://github.com/ailever/ailever/tree/master/ailever/dataset)
- [EDA-BasicDataAnalysis](https://github.com/ailever/ailever/blob/master/ailever/analysis/eda-basic_data_analysis.md)
- [EDA-DecisionTree](https://github.com/ailever/ailever/blob/master/ailever/analysis/eda-decision_tree.md)
- [EDA-FactorAnalysis](https://github.com/ailever/ailever/blob/master/ailever/analysis/eda-factor_analysis.md)
- [EDA-FeatureImportance](https://github.com/ailever/ailever/blob/master/ailever/analysis/eda-feature_importance.md)
- [EDA-Visualization](https://github.com/ailever/ailever/blob/master/ailever/analysis/eda-visualization.md)
- [Preprocessing-MissingValueImputation](https://github.com/ailever/ailever/blob/master/ailever/analysis/prep-missing_value_imputation.md)
- [Preprocessing-OutlierDetection](https://github.com/ailever/ailever/blob/master/ailever/analysis/prep-outlier_detection.md)
- [Preprocessing-DimensionalityReduction](https://github.com/ailever/ailever/blob/master/ailever/analysis/prep-dimensionality_reduction.md)
- [Preprocessing-FeatureExtraction](https://github.com/ailever/ailever/blob/master/ailever/analysis/prep-feature_extraction.md)

---

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
pd.set_option('display.max_rows', 10)
plt.rcParams["font.family"] = 'NanumBarunGothic'
```

## REVIEW
### Tabular Data Preparation
- [Duplication Row]/[Single Value Column]/[Low Variance Column] 'removal'/'keep'
- [Missising Value] 'removal'/'imputation'
- Column-based dimensionality 'reduction'/'enlargement'
    - Conceptual data dimensionality reduction: statistical information-based filter
    - Mathematical data dimensionality reduction
    - 'RFE'/'Sequential feature selector'
    - Feature generation: polynomial feature transform
- Row-based dimensionality 'reduction'/'enlargement'
    - Data having imbalanced class sampling
    - Cost-sensitive algorithms
    - One-class algorithms
    - Probability tuning algorithms
- Stability improvement
    - Data scaling
    - Discretization transform
    - Make uniform-like
    - Make gaussian-like



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

### Table Definition
```python
from ailever.dataset import UCI
from ailever.analysis import EDA

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)
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
```python
from ailever.dataset import UCI
from ailever.analysis import EDA
from ailever.analysis import DataTransformer

frame = UCI.adult(download=False)
frame['fnlwgt'] = DataTransformer.ef_binning(frame[['fnlwgt']].astype(int), target_columns='fnlwgt', bins=[40,120], only_transform=True)['fnlwgt_ef120bins']

eda = EDA(frame, verbose=False)
eda.cleaning(as_int=['age', 'education-num', 'capital-loss', 'capital-gain', 'hours-per-week', 'fnlwgt'])
eda.attributes_specification(visual_on=True)
```
```python
from pandas.plotting import scatter_matrix
from ailever.dataset import UCI
from ailever.analysis import EDA

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)
eda.cleaning(as_int=['age', 'hours-per-week', 'fnlwgt'], as_float=['capital-gain', 'capital-loss', 'education-num'], as_str=all, verbose=False)
eda.plot()
```


### Exploratory Numerical Variable Analysis
#### Visualization
```python
from pandas.plotting import scatter_matrix
from ailever.dataset import SKAPI

frame = SKAPI.housing(download=False)
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
```
```python
from pandas.plotting import scatter_matrix
from ailever.dataset import UCI
from ailever.analysis import EDA

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)
eda.cleaning(as_int=['age'], as_float=['capital-gain', 'education-num'], as_str=all, verbose=False)
eda.frame[eda.integer_columns+eda.float_columns].hist(layout=(1,3), figsize=(25,5), edgecolor='white')
eda.frame[eda.integer_columns+eda.float_columns].plot(kind='density', subplots=True, layout=(1,3), figsize=(25,5))
eda.frame[eda.integer_columns+eda.float_columns].plot(kind='box', subplots=True, layout=(1,3), figsize=(25,5))
scatter_matrix(eda.frame[eda.integer_columns+eda.float_columns], figsize=(25,15), hist_kwds=dict(edgecolor='white'))
```

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
#### Visualization
```python
from ailever.dataset import UCI
from ailever.analysis import EDA

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)
eda.cleaning(as_int=['age', 'hours-per-week', 'fnlwgt'], as_float=['capital-loss', 'capital-gain', 'education-num'], as_str=all, verbose=False)
eda.plot()
```

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
eda.information_values(target_column='age', visual_on='EventIVSum') # Opts] visual_on : 'EventIVSum', 'EventIVAvg', 'QuasiBVF'
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
```python
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=3000, n_features=25, n_informative=4, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1)

model = XGBClassifier(eval_metric='mlogloss')
model.fit(X, y)

_, axes = plt.subplots(figsize=(25,7*max(int(X.shape[1] / 20), 1)))
plot_importance(model, max_num_features=20, ax=axes, show_values=True)
model.feature_importances_
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

<br><br><br>

---

## Data-Preprocessing : DataTransformer
### Basic Data Transformation

`time_splitor`
```python
from ailever.analysis import DataTransformer
from ailever.dataset import SMAPI

frame = SMAPI.co2(download=False)
frame['date'] = frame.index
DataTransformer.time_splitor(frame)
```

`temporal_smoothing`
```python
from ailever.analysis import DataTransformer
from ailever.dataset import SMAPI

frame = SMAPI.co2(download=False).dropna().reset_index().rename(columns={'index':'date'})
DataTransformer.temporal_smoothing(frame, target_column='co2', date_column='date', freq='D', smoothing_order=1, decimal=1, including_model_object=False, only_transform=False, keep=True)
```

`spatial_smoothing`
```python
from ailever.analysis import DataTransformer
from ailever.dataset import SMAPI

frame = SMAPI.co2(download=False).dropna()
DataTransformer.spatial_smoothing(frame, target_column='co2', only_transform=True, windows=[5,10,20,30], stability_feature=False)
```

`sequence_parallelizing`
```python
from ailever.analysis import DataTransformer
from ailever.dataset import SMAPI

frame = SMAPI.co2(download=False)
DataTransformer.sequence_parallelizing(frame, target_column='co2', only_transform=False, keep=False, window=5)
```

`to_numeric`
```python
from ailever.dataset import UCI
from ailever.analysis import DataTransformer

frame = UCI.adult(download=False)
DataTransformer.to_numeric(frame, target_column='education', epochs=1000, num_feature=5, only_transform=True)
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


`ef_binning`
```python
from ailever.dataset import UCI
from ailever.analysis import DataTransformer

frame = UCI.adult(download=False)
#DataTransformer.empty()
#DataTransformer.build()
DataTransformer.ef_binning(frame, target_columns=['age', 'fnlwgt'], bins=[4], only_transform=True, keep=False) # Check DataTransformer.storage_box[-1] when keep == True
```

`abs_diff`
```python
from ailever.analysis import DataTransformer
from ailever.dataset import SMAPI

frame = SMAPI.co2(download=False)
DataTransformer.abs_diff(frame, target_columns=['co2'], only_transform=False, keep=False, binary=False, periods=[10,20,30], within_order=2)
```

`rel_diff`
```python
from ailever.analysis import DataTransformer
from ailever.dataset import SMAPI

frame = SMAPI.co2(download=False)
DataTransformer.rel_diff(frame, target_columns=['co2'], only_transform=False, keep=False, binary=False, periods=[10,20,30], within_order=2)
```


### Advanced Data Transformation
`time_splitor`
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
frame = DataTransformer.abs_diff(frame, target_columns=['co2'], only_transform=False, keep=False, binary=True, periods=2)

eda = EDA(frame, verbose=False)
eda.information_values(target_column='co2_absderv1st2', target_event=1)
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
frame = DataTransformer.abs_diff(frame, target_columns=['co2'], only_transform=False, keep=False, binary=True, periods=2)

eda = EDA(frame, verbose=False)
eda.information_values(target_column='co2_absderv1st2', target_event=1)
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


`differencing`
```python
from ailever.analysis import EDA, DataTransformer
from ailever.dataset import SMAPI

frame = SMAPI.co2(download=False)
frame = DataTransformer.abs_diff(frame, target_columns=['co2'], only_transform=False, keep=False, binary=True, periods=[2, 10,20,30], within_order=2)
frame = DataTransformer.rel_diff(frame, target_columns=['co2'], only_transform=False, keep=False, binary=False, periods=[2, 10,20,30], within_order=2)

eda = EDA(frame, verbose=False)
eda.information_values(target_column='co2_absderv1st10')
```




### Data Cleaning


<br><br><br>

---


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
frame = DataTransformer.temporal_smoothing(frame, target_column='co2', date_column='date', freq='D', smoothing_order=1, decimal=1, including_model_object=False, only_transform=False, keep=True)
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

---

## Evaluation
`Summary`
```python
from ailever.analysis import Evaluation

Evaluation.target_class_evaluation(y_true, y_pred)
Evaluation.roc_curve(y_true, y_prob)
Evaluation.pr_curve(y_true, y_prob)
Evaluation.feature_importance(X_true, y_true)
Evaluation.permutation_importance(X_true, y_true)
Evaluation.information_value(X_true, y_true)
Evaluation.decision_tree(table) # X, y_true
Evaluation.imputation(X, new_X)
Evaluation.imbalance(X, new_X)
Evaluation.linearity(X_true, y_true)

Evaluation('classification').about(y_true, y_pred)
Evaluation('regression').about(y_true, y_pred)
Evaluation('clustering').about(X, new_X)
Evaluation('manifolding').about(X, new_X)
Evaluation('filtering').about(observed_features, filtered_features)
```

`target_class_evaluation`
```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from ailever.analysis import Evaluation

X, y = make_classification(n_samples=300, n_features=8, n_informative=5, n_redundant=1, n_repeated=1, n_classes=6, n_clusters_per_class=1, weights=[1/10, 3/10, 2/10, 1/10, 3/10])
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_pred = classifier.predict(X)
Evaluation.target_class_evaluation(y_true, y_pred)
```

`roc_curve: predicted_condition=True`
```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from ailever.analysis import Evaluation

X, y = make_classification(n_samples=3000, n_features=8, n_informative=5, n_redundant=1, n_repeated=1, n_classes=3, n_clusters_per_class=1)
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_prob = classifier.predict_proba(X)

FPR_TPRs, AUCs = Evaluation.roc_curve(y_true, y_prob, num_threshold=21, predicted_condition=True, visual_on=False)
for (target_class, FPR_TPR), AUC in zip(FPR_TPRs.items(), AUCs.values()):
    plt.plot(FPR_TPR.loc['FPR'].values, FPR_TPR.loc['TPR'].values, marker='o', label=str(target_class)+' | '+str(round(AUC, 2)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Fall-Out')
plt.ylabel('Recall')
plt.legend()
```

`roc_curve: predicted_condition=False`
```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from ailever.analysis import Evaluation

X, y = make_classification(n_samples=300, n_features=8, n_informative=2, n_redundant=1, n_repeated=1, n_classes=3, n_clusters_per_class=1, weights=[0.1,0.1,0.8])
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_prob = classifier.predict_proba(X)

FNR_TNRs, AUCs = Evaluation.roc_curve(y_true, y_prob, num_threshold=11, predicted_condition=False, visual_on=False)
for (target_class, FNR_TNR), AUC in zip(FNR_TNRs.items(), AUCs.values()):
    plt.plot(FNR_TNR.loc['FNR'].values, FNR_TNR.loc['TNR'].values, marker='o', label=str(target_class)+' | '+str(round(AUC, 2)))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Miss-Rate')
plt.ylabel('Selectivity')
plt.legend()
```

`roc_curve: predicted_condition=None(default)`
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from ailever.analysis import Evaluation

X, y = make_classification(n_samples=300, n_features=8, n_informative=2, n_redundant=1, n_repeated=1, n_classes=3, n_clusters_per_class=1, weights=[0.1,0.1,0.8])
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_prob = classifier.predict_proba(X)

(FPR_TPRs, P_AUCs), (FNR_TNRs, N_AUCs) = Evaluation.roc_curve(y_true, y_prob, num_threshold=11, visual_on=False)

target_class = np.unique(y)[-1]
FPR_TPRs[target_class]
FNR_TNRs[target_class]
P_AUCs[target_class]
N_AUCs[target_class]
```

`pr_curve: predicted_condition=True`
```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from ailever.analysis import Evaluation

X, y = make_classification(n_samples=300, n_features=8, n_informative=2, n_redundant=1, n_repeated=1, n_classes=3, n_clusters_per_class=1, weights=[0.1,0.1,0.8])
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_prob = classifier.predict_proba(X)

PPV_TPRs, AUCs = Evaluation.pr_curve(y_true, y_prob, num_threshold=11, predicted_condition=True, visual_on=False)
for (target_class, PPV_TPR), AUC in zip(PPV_TPRs.items(), AUCs.values()):
    plt.plot(PPV_TPR.loc['TPR'].values, PPV_TPR.loc['PPV'].values, marker='o', label=str(target_class)+' | '+str(round(AUC, 2)))
plt.plot([0, 1], [1, 0], 'k--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
```

`pr_curve: predicted_condition=False`
```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from ailever.analysis import Evaluation

X, y = make_classification(n_samples=300, n_features=8, n_informative=2, n_redundant=1, n_repeated=1, n_classes=3, n_clusters_per_class=1, weights=[0.1,0.1,0.8])
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_prob = classifier.predict_proba(X)

NPV_TNRs, AUCs = Evaluation.pr_curve(y_true, y_prob, num_threshold=11, predicted_condition=False, visual_on=False)
for (target_class, NPV_TNR), AUC in zip(NPV_TNRs.items(), AUCs.values()):
    plt.plot(NPV_TNR.loc['TNR'].values, NPV_TNR.loc['NPV'].values, marker='o', label=str(target_class)+' | '+str(round(AUC, 2)))
plt.plot([0, 1], [1, 0], 'k--')
plt.xlabel('Selectivity')
plt.ylabel('NegativePredictiveValue')
plt.legend()
```

`pr_curve: predicted_condition=None(default)`
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from ailever.analysis import Evaluation

X, y = make_classification(n_samples=3000, n_features=8, n_informative=4, n_redundant=1, n_repeated=1, n_classes=5, n_clusters_per_class=1, weights=[0.1, 0.1, 0.8, 0.1, 0.1])
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_prob = classifier.predict_proba(X)

(PPV_TPRs, P_AUCs), (NPV_TNRs, N_AUCs)  = Evaluation.pr_curve(y_true, y_prob, num_threshold=11, visual_on=True)

target_class = np.unique(y)[-1]
PPV_TPRs[target_class]
NPV_TNRs[target_class]
P_AUCs[target_class]
N_AUCs[target_class]
```

`decision_tree`
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from ailever.analysis import Evaluation

X, y = make_classification(n_samples=3000, n_features=4, n_informative=4, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1)
table = pd.DataFrame(data=np.c_[X, y], columns = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'] + ['target'])
table.attrs['target_names'] = ['setosa', 'versicolor', 'virginica']

Evaluation.decision_tree(table, min_samples_leaf=100, min_samples_split=30, max_depth=3)
```

`feature_importance`
```python
from ailever.analysis import Evaluation
from sklearn.datasets import make_classification, make_regression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

linear_model = LinearRegression()
nonlinear_model = XGBRegressor()

#X, y = make_classification(n_samples=3000, n_features=25, n_informative=4, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1)
X, y = make_regression(n_samples=3000, n_features=10, n_informative=5, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)

Evaluation.feature_property(linear_model, X, y, permutation=True, visual_on=True)
Evaluation.feature_property(nonlinear_model, X, y, permutation=True, visual_on=True)
```

```python
from ailever.analysis import Evaluation
Evaluation('classification').about(y_true, y_pred)
```

```python
from ailever.analysis import Evaluation
Evaluation('regression').about(y_true, y_pred)
```

```python
from ailever.analysis import Evaluation
Evaluation('clustering').about(X)
```

```python
from ailever.analysis import Evaluation
Evaluation('imputation').about()
Evaluation('imbalance').about()
Evaluation('reduction').about()
Evaluation('linearity').about()
```



---

## Probability
```python
from ailever.analysis import Probability

probability = Probability(params=dict(trial=30, expected_occurence=20, success_probability=2/3, life_time=10))
probability
```
