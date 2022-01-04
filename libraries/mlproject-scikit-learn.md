
### Classification
```python
from functools import wraps
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from ailever.dataset import SKAPI
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import linear_model, neighbors, tree
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn import metrics

class TemplateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

def evaluation(*args, **kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(model, X, y, verbose=False): 
            y, y_pred = func(model, X, y) 
            if verbose:
                print(classification_report(y, y_pred))                
            summary = dict()
            summary['ACC'] = [metrics.accuracy_score(y, y_pred)]
            summary['PPV'] = [metrics.precision_score(y, y_pred, average='micro')]    
            summary['TPR'] = [metrics.recall_score(y, y_pred, average='micro')]
            summary['F1'] = [metrics.f1_score(y, y_pred, average='micro')]
            evaluation = pd.DataFrame(summary)
            return y_pred, evaluation
        return wrapper
    return decorator

@evaluation(description="my_description")
def pred_evaluation(model, X, y, verbose=False):
    y_pred = model.predict(X)
    return y, y_pred

dataset = SKAPI.digits(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

feature_space = FeatureUnion(transformer_list=[('TemplateTransformer', TemplateTransformer()),])

pipelines = dict()
pipelines['KNeighborsClassifier'] = Pipeline(steps=[('FeatureSpace', feature_space), ('KNeighborsClassifier', neighbors.KNeighborsClassifier())])
pipelines['ExtraTreeClassifier'] = Pipeline(steps=[('FeatureSpace', feature_space), ('ExtraTreeClassifier', tree.ExtraTreeClassifier())])

# syntax: <estimator>__<parameter>
param_grids = dict()
param_grids['KNeighborsClassifier'] = dict(
    KNeighborsClassifier__weights = ['uniform', 'distance']
)
param_grids['ExtraTreeClassifier'] = dict(
    ExtraTreeClassifier__criterion = ['gini', 'entropy']
)

results = []
names = []
# https://scikit-learn.org/stable/modules/model_evaluation.html
for idx, ((name, pipeline), param_grid) in enumerate(zip(pipelines.items(), param_grids.values())):
    scorings = ['accuracy']
    scoring = scorings[0]
    cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    pipeline = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cross_validation, scoring=scoring)
    pipeline.fit(X, y)
    #pipeline.cv_results_
    #pipeline.best_params_
    #pipeline.best_estimator_
    #pipeline.best_score_

    # [STEP3]: save & load
    joblib.dump(pipeline, f'{idx}{name}_pipeline.joblib')
    pipeline = joblib.load(f'{idx}{name}_pipeline.joblib')

    # [STEP4]: prediction & evaluation
    y_pred, evaluation = pred_evaluation(pipeline, X, y)
    eval_table = evaluation if idx == 0 else eval_table.append(evaluation)
        
    #print('*', name)
    #print(metrics.classification_report(y, y_pred))

    names.append(name)
    results.append(cross_val_score(pipeline, X, y, cv=cross_validation, scoring=scoring))
    
fig = plt.figure(figsize=(25,7)); layout=(1,1); axes = dict()
axes[0] = plt.subplot2grid(layout, (0,0), fig=fig)
axes[0].boxplot(results)
axes[0].set_title('Evaluate Algorithms')
axes[0].set_xticklabels(names)
plt.show()

eval_table
```

### Regression
```python
from functools import wraps
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, KFold
from sklearn import metrics

class TemplateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import pandas as pd
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        features_by_vif = pd.Series(
            data = [variance_inflation_factor(X, i) for i in range(X.shape[1])], 
            index = range(X.shape[1])).sort_values(ascending=True).iloc[:X.shape[1] - 1].index.tolist()
        return X[:, features_by_vif]

def evaluation(*args, **kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(model, X, y, verbose=False): 
            y, y_pred = func(model, X, y) 
            if verbose:
                print(classification_report(y, y_pred))                
            summary = dict()
            summary['MAE'] = [metrics.mean_absolute_error(y, y_pred)]
            summary['MAPE'] = [metrics.mean_absolute_percentage_error(y, y_pred)]
            summary['MSE'] = [metrics.mean_squared_error(y, y_pred)]    
            summary['R2'] = [metrics.r2_score(y, y_pred)]
            evaluation = pd.DataFrame(summary)
            return y_pred, evaluation
        return wrapper
    return decorator

@evaluation(description="my_description")
def pred_evaluation(model, X, y, verbose=False):
    y_pred = model.predict(X)
    return y, y_pred
    
X, y = make_regression(n_samples=3000, n_features=10, n_informative=5, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)
feature_space = FeatureUnion(transformer_list=[('TemplateTransformer', TemplateTransformer()),])

pipelines = dict()
pipelines['LinearRegression'] = Pipeline(steps=[('FeatureSpace', feature_space), ('LinearRegression', LinearRegression())])

# syntax: <estimator>__<parameter>
param_grids = dict()
param_grids['LinearRegression'] = dict(
    LinearRegression__fit_intercept = [False, True]
)

# https://scikit-learn.org/stable/modules/model_evaluation.html
names = []
results = []
for idx, ((name, pipeline), param_grid) in enumerate(zip(pipelines.items(), param_grids.values())):
    scorings = ['neg_mean_squared_error']
    scoring = scorings[0]
    
    cross_validation = KFold(n_splits=10, shuffle=True, random_state=None)
    pipeline = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cross_validation, scoring=scoring)
    pipeline.fit(X, y)

    # [STEP3]: save & load
    joblib.dump(pipeline, f'{idx}{name}_pipeline.joblib')
    pipeline = joblib.load(f'{idx}{name}_pipeline.joblib')

    # [STEP4]: prediction & evaluation
    y_pred, evaluation = pred_evaluation(pipeline, X, y)
    eval_table = evaluation if idx == 0 else eval_table.append(evaluation)
        
    #print('*', name)
    names.append(name)
    results.append(cross_val_score(pipeline, X, y, cv=cross_validation, scoring=scoring))

fig = plt.figure(figsize=(25,7)); layout=(1,1); axes = dict()
axes[0] = plt.subplot2grid(layout, (0,0), fig=fig)
axes[0].boxplot(results)
axes[0].set_title('Evaluate Algorithms')
axes[0].set_xticklabels(names)
plt.show()

eval_table
```
