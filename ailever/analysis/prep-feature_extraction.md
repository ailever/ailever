## Single Transform-based Feature Extraction
### (RFE)Manually, Select the number of features
```python
# data preparation as feature engineering with feature selection for wine dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, KBinsDiscretizer
from ailever.dataset import SKAPI

dataset = SKAPI.wine(download=False)
dataset['od280/od315_of_diluted_wines'] = dataset['od280/od315_of_diluted_wines'].round()
X = dataset.loc[:, dataset.columns != 'od280/od315_of_diluted_wines'].values
y = dataset.loc[:, dataset.columns == 'od280/od315_of_diluted_wines'].values.ravel()

# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))

# transforms for the feature union
transforms = list()
transforms.append(('MinMaxScaler', MinMaxScaler()))
transforms.append(('StandardScaler', StandardScaler()))
transforms.append(('RobustScaler', RobustScaler()))
transforms.append(('QuantileTransformer', QuantileTransformer(n_quantiles=100, output_distribution='normal')))
transforms.append(('KBinsDiscretizer', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')))
transforms.append(('PCA', PCA(n_components=7)))
transforms.append(('TruncatedSVD', TruncatedSVD(n_components=7)))

cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
feature_selector = RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=15)
model = LogisticRegression(solver='liblinear')

names = list()
metrics = dict()
metrics['accuracy'] = list()
metrics['precision_micro'] = list()
metrics['recall_micro'] = list()
metrics['f1_micro'] = list()
for transform in transforms:
    steps = list()
    steps.append(('feature_space', FeatureUnion([transform])))
    steps.append(('feature_selector', feature_selector))
    steps.append(('model', model))
    
    pipeline = Pipeline(steps=steps)
    #pipeline = Pipeline(steps=steps).fit(X, y)
    #pipeline.support_
    #pipeline.ranking_
    
    names.append(transform[0])
    for metric_name in metrics.keys():
        metrics[metric_name].append(cross_val_score(pipeline, X, y, scoring=metric_name, cv=cross_validation, n_jobs=-1))
        
fig = plt.figure(figsize=(25,5*len(metrics))); layout=(len(metrics), 1); axes = dict()
for idx, metric_name in enumerate(metrics.keys()):
    axes[idx] = plt.subplot2grid(layout, (idx,0), fig=fig)
    axes[idx].boxplot(metrics[metric_name])
    axes[idx].set_title(metric_name)
    axes[idx].set_xticklabels(names)
plt.show()

metric_df = pd.DataFrame(columns=['Metric'] + names)
for metric_name in metrics.keys():
    df = pd.DataFrame(data=np.c_[metrics[metric_name]].T, columns=names)
    df['Metric'] = metric_name
    metric_df = metric_df.append(df)
metric_df.reset_index().rename(columns={'index':'IterNum'}).set_index(['Metric', 'IterNum'])
```

### (RFECV)Automatically, Select the number of features
```python
# data preparation as feature engineering with feature selection for wine dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, KBinsDiscretizer
from ailever.dataset import SKAPI

dataset = SKAPI.wine(download=False)
dataset['od280/od315_of_diluted_wines'] = dataset['od280/od315_of_diluted_wines'].round()
X = dataset.loc[:, dataset.columns != 'od280/od315_of_diluted_wines'].values
y = dataset.loc[:, dataset.columns == 'od280/od315_of_diluted_wines'].values.ravel()

# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))

# transforms for the feature union
transforms = list()
transforms.append(('MinMaxScaler', MinMaxScaler()))
transforms.append(('StandardScaler', StandardScaler()))
transforms.append(('RobustScaler', RobustScaler()))
transforms.append(('QuantileTransformer', QuantileTransformer(n_quantiles=100, output_distribution='normal')))
transforms.append(('KBinsDiscretizer', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')))
transforms.append(('PCA', PCA(n_components=7)))
transforms.append(('TruncatedSVD', TruncatedSVD(n_components=7)))

cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
feature_selector = RFECV(estimator=LogisticRegression(solver='liblinear'), cv=cross_validation)
model = LogisticRegression(solver='liblinear')

names = list()
metrics = dict()
metrics['accuracy'] = list()
metrics['precision_micro'] = list()
metrics['recall_micro'] = list()
metrics['f1_micro'] = list()
for transform in transforms:
    steps = list()
    steps.append(('feature_space', FeatureUnion([transform])))
    steps.append(('feature_selector', feature_selector))
    steps.append(('model', model))
    pipeline = Pipeline(steps=steps)
    #pipeline = Pipeline(steps=steps).fit(X, y)
    #pipeline.support_
    #pipeline.ranking_

    names.append(transform[0])
    for metric_name in metrics.keys():
        metrics[metric_name].append(cross_val_score(pipeline, X, y, scoring=metric_name, cv=cross_validation, n_jobs=-1))
        
fig = plt.figure(figsize=(25,5*len(metrics))); layout=(len(metrics), 1); axes = dict()
for idx, metric_name in enumerate(metrics.keys()):
    axes[idx] = plt.subplot2grid(layout, (idx,0), fig=fig)
    axes[idx].boxplot(metrics[metric_name])
    axes[idx].set_title(metric_name)
    axes[idx].set_xticklabels(names)
plt.show()

metric_df = pd.DataFrame(columns=['Metric'] + names)
for metric_name in metrics.keys():
    df = pd.DataFrame(data=np.c_[metrics[metric_name]].T, columns=names)
    df['Metric'] = metric_name
    metric_df = metric_df.append(df)
metric_df.reset_index().rename(columns={'index':'IterNum'}).set_index(['Metric', 'IterNum'])
```

## FeatureUnion-based Feature Extraction
### (RFE)Manually, Select the number of features
```python
# data preparation as feature engineering with feature selection for wine dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, KBinsDiscretizer
from ailever.dataset import SKAPI

dataset = SKAPI.wine(download=False)
dataset['od280/od315_of_diluted_wines'] = dataset['od280/od315_of_diluted_wines'].round()
X = dataset.loc[:, dataset.columns != 'od280/od315_of_diluted_wines'].values
y = dataset.loc[:, dataset.columns == 'od280/od315_of_diluted_wines'].values.ravel()

# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))

# transforms for the feature union
feature_spaces = dict()
feature_spaces[0] = list()
feature_spaces[0].append(('StandardScaler', StandardScaler()))
feature_spaces[0].append(('QuantileTransformer', QuantileTransformer(n_quantiles=100, output_distribution='normal')))
feature_spaces[0].append(('PCA', PCA(n_components=7)))
feature_spaces[1] = list()
feature_spaces[1].append(('RobustScaler', RobustScaler()))
feature_spaces[1].append(('KBinsDiscretizer', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')))
feature_spaces[1].append(('TruncatedSVD', TruncatedSVD(n_components=7)))

cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
feature_selector = RFE(estimator=LogisticRegression(solver='liblinear'), n_features_to_select=15)
model = LogisticRegression(solver='liblinear')

names = list(map(lambda x: 'FeatureSpace_'+str(x), feature_spaces.keys()))
metrics = dict()
metrics['accuracy'] = list()
metrics['precision_micro'] = list()
metrics['recall_micro'] = list()
metrics['f1_micro'] = list()
for feature_space in feature_spaces.values():
    steps = list()
    steps.append(('feature_space', FeatureUnion(feature_space)))
    steps.append(('feature_selector', feature_selector))
    steps.append(('model', model))
    
    pipeline = Pipeline(steps=steps)
    #pipeline = Pipeline(steps=steps).fit(X, y)
    #pipeline.support_
    #pipeline.ranking_

    for metric_name in metrics.keys():
        metrics[metric_name].append(cross_val_score(pipeline, X, y, scoring=metric_name, cv=cross_validation, n_jobs=-1))
        
fig = plt.figure(figsize=(25,5*len(metrics))); layout=(len(metrics), 1); axes = dict()
for idx, metric_name in enumerate(metrics.keys()):
    axes[idx] = plt.subplot2grid(layout, (idx,0), fig=fig)
    axes[idx].boxplot(metrics[metric_name])
    axes[idx].set_title(metric_name)
    axes[idx].set_xticklabels(names)
plt.show()

metric_df = pd.DataFrame(columns=['Metric'] + names)
for metric_name in metrics.keys():
    df = pd.DataFrame(data=np.c_[metrics[metric_name]].T, columns=names)
    df['Metric'] = metric_name
    metric_df = metric_df.append(df)
metric_df.reset_index().rename(columns={'index':'IterNum'}).set_index(['Metric', 'IterNum'])
```

### (RFECV)Automatically, Select the number of features
```python
# data preparation as feature engineering with feature selection for wine dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, KBinsDiscretizer
from ailever.dataset import SKAPI

dataset = SKAPI.wine(download=False)
dataset['od280/od315_of_diluted_wines'] = dataset['od280/od315_of_diluted_wines'].round()
X = dataset.loc[:, dataset.columns != 'od280/od315_of_diluted_wines'].values
y = dataset.loc[:, dataset.columns == 'od280/od315_of_diluted_wines'].values.ravel()

# minimally prepare dataset
X = X.astype('float')
y = LabelEncoder().fit_transform(y.astype('str'))

# transforms for the feature union
feature_spaces = dict()
feature_spaces[0] = list()
feature_spaces[0].append(('StandardScaler', StandardScaler()))
feature_spaces[0].append(('QuantileTransformer', QuantileTransformer(n_quantiles=100, output_distribution='normal')))
feature_spaces[0].append(('PCA', PCA(n_components=7)))
feature_spaces[1] = list()
feature_spaces[1].append(('RobustScaler', RobustScaler()))
feature_spaces[1].append(('KBinsDiscretizer', KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')))
feature_spaces[1].append(('TruncatedSVD', TruncatedSVD(n_components=7)))

cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
feature_selector = RFECV(estimator=LogisticRegression(solver='liblinear'))
model = LogisticRegression(solver='liblinear')

names = list(map(lambda x: 'FeatureSpace_'+str(x), feature_spaces.keys()))
metrics = dict()
metrics['accuracy'] = list()
metrics['precision_micro'] = list()
metrics['recall_micro'] = list()
metrics['f1_micro'] = list()
for feature_space in feature_spaces.values():
    steps = list()
    steps.append(('feature_space', FeatureUnion(feature_space)))
    steps.append(('feature_selector', feature_selector))
    steps.append(('model', model))
    
    pipeline = Pipeline(steps=steps)
    #pipeline = Pipeline(steps=steps).fit(X, y)
    #pipeline.support_
    #pipeline.ranking_

    for metric_name in metrics.keys():
        metrics[metric_name].append(cross_val_score(pipeline, X, y, scoring=metric_name, cv=cross_validation, n_jobs=-1))
        
fig = plt.figure(figsize=(25,5*len(metrics))); layout=(len(metrics), 1); axes = dict()
for idx, metric_name in enumerate(metrics.keys()):
    axes[idx] = plt.subplot2grid(layout, (idx,0), fig=fig)
    axes[idx].boxplot(metrics[metric_name])
    axes[idx].set_title(metric_name)
    axes[idx].set_xticklabels(names)
plt.show()

metric_df = pd.DataFrame(columns=['Metric'] + names)
for metric_name in metrics.keys():
    df = pd.DataFrame(data=np.c_[metrics[metric_name]].T, columns=names)
    df['Metric'] = metric_name
    metric_df = metric_df.append(df)
metric_df.reset_index().rename(columns={'index':'IterNum'}).set_index(['Metric', 'IterNum'])
```
