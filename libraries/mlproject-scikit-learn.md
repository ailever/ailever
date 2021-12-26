
### Classification
```python
from functools import wraps
import joblib
import matplotlib.pyplot as plt
from ailever.dataset import SKAPI
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import linear_model, neighbors, tree
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import metrics

def evaluation(*args, **kwargs):
    def decorator(func):
        @wraps(func)
        def wrapper(model, X, y, verbose=False): 
            y, y_pred = func(model, X, y) 
            if verbose:
                print(classification_report(y, y_pred))                
            summary = dict()
            summary['accuracy_score'] = [metrics.accuracy_score(y, y_pred)]
            summary['precision_score'] = [metrics.precision_score(y, y_pred, average='micro')]    
            summary['recall_score'] = [metrics.recall_score(y, y_pred, average='micro')]
            summary['f1_score'] = [metrics.f1_score(y, y_pred, average='micro')]
            evaluation = pd.DataFrame(summary)
            return y_pred, evaluation
        return wrapper
    return decorator

@evaluation(description="my_description")
def pred_evaluation(model, X, y, verbose=False):
    y_pred = classifier.predict(X)
    return y, y_pred


dataset = SKAPI.digits(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

pipelines = dict()
pipelines['KNeighborsClassifier'] = Pipeline(steps=[('KNeighborsClassifier', neighbors.KNeighborsClassifier())])
pipelines['ExtraTreeClassifier'] = Pipeline(steps=[('ExtraTreeClassifier', tree.ExtraTreeClassifier())])

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
for idx, ((name, pipeline), param_grid) in enumerate(zip(pipelines.items(), param_grids.values())):
    scorings = ['accuracy']
    scoring = scorings[0]
    cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    classifier = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cross_validation, scoring=scoring)
    classifier.fit(X, y)
    #classifier.cv_results_
    #classifier.best_params_
    #classifier.best_estimator_
    #classifier.best_score_

    # [STEP3]: save & load
    joblib.dump(classifier, f'{idx}{name}_classifier.joblib')
    classifier = joblib.load(f'{idx}{name}_classifier.joblib')

    # [STEP4]: prediction & evaluation
    y_pred, evaluation = pred_evaluation(classifier, X, y)
    eval_table = evaluation if idx == 0 else eval_table.append(evaluation)
        
    #print('*', name)
    #print(classification_report(y, y_pred))

    names.append(name)
    results.append(cross_val_score(classifier, X, y, cv=cross_validation, scoring=scoring))
    
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
```
