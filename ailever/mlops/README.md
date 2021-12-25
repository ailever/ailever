## Supervised Learning
### inference
```python
from ailever.mlops import project
from ailever.dataset import SKAPI
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.linear_model import LogisticRegression
import xgboost
import lightgbm
import catboost

dataset0 = SKAPI.iris(download=False)
dataset1 = SKAPI.digits(download=False)

model0 = ExtraTreesClassifier()
model1 = LogisticRegression()
model2 = xgboost.XGBClassifier()
model3 = lightgbm.LGBMClassifier()
model4 = catboost.CatBoostClassifier()


mlops = project({
    'root':'my_proeject',
    'feature_store':'my_fs', 
    'model_registry':'my_mr', 
    'source_repository':'my_sr', 
    'metadata_store':'my_ms'})

mlops.dataset = [dataset0, (dataset1, 'd_comment1')]
mlops.model = [model0, model1, model2, (model3, 't_comment3'), (model4, 't_comment4')]
mlops.feature_choice(0).model_choice(1)  # if not call choice functions, last things(-1) is always selected.
#mlops.dataset, mlops.model  # dataset, model from memory

mlops.training_board() # mlops.training_board(log='inside')
mlops.inference(dataset0, mode='evaluation', verbose=False)
mlops.inference(dataset0, mode='visualization', verbose=False)
mlops.inference(dataset0, mode='prediction', verbose=False)
mlops.prediction(dataset0.loc[:10, dataset0.columns!='target']) 
```
```python
mlops.PredictResult.cls_evaluation(mlops.dataset['target'], mlops.prediction())

model = mlops.drawup_model('20211219_123402-LGBMClassifier.joblib')  # from model_registry
dataset = mlops.drawup_dataset('20211219_123400-dataset0.csv')       # from feature_store
mlops.PredictResult.cls_evaluation(dataset['target'], model.prediction(dataset.loc[:, dataset.columns !='target']))
```


### storing_model
```python
from ailever.mlops import project
from ailever.dataset import SKAPI
from sklearn.ensemble import ExtraTreesClassifier 

dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, 'target'].ravel()
        
model = ExtraTreesClassifier()
model.fit(X, y)

mlops = project({
    'root':'my_proeject',
    'feature_store':'my_fs', 
    'model_registry':'my_mr', 
    'source_repository':'my_sr', 
    'metadata_store':'my_ms'})

mlops.storing_model(model, comment='my_model')
#mlops.model_choice('20211220_005107-CatBoostRegressor.joblib')
mlops.prediction(dataset.loc[:10, dataset.columns!='target'])
mlops.inference(dataset, mode='evaluation', verbose=False)
mlops.inference(dataset, mode='visualization', verbose=False)
mlops.inference(dataset, mode='prediction', verbose=False)
mlops.training_board(log='outside')
```


### codecommit

```python
%%writefile my_code.py

from ailever.dataset import SKAPI
from sklearn.ensemble import ExtraTreesClassifier 
import matplotlib.pyplot as plt

## Structured Functions
def preprocessing():
    # DEFINE mlops.dataset
    dataset = SKAPI.iris(download=False)
    return dataset

def architecture():
    # DEFINE mlops.model
    model = ExtraTreesClassifier()
    return model

def train(model, dataset):
    # DEFINE mlops.[framework].train
    X = dataset.loc[:, dataset.columns != 'target']
    y = dataset.loc[:, 'target'].ravel()
    model.fit(X, y)
    return model


## Unstructured Functions
def predict(model, X):
    pred_val = model.predict(X)
    plt.plot(pred_val)
    return pred_val

def evaluate(y, pred_val):
    metric = ((y - pred_val)**2).sum()
    return metric

def report(metric):
    report = metric
    return report    
```

```python
from ailever.mlops import project
        
mlops = project({
    'root':'my_proeject',
    'feature_store':'my_fs', 
    'model_registry':'my_mr', 
    'source_repository':'my_sr', 
    'metadata_store':'my_ms'})

mlops.codecommit(entry_point='my_code.py')
mlops.prediction(slice(0,10,1)) # inference for last dataset and model 
mlops.inference(slice(0,10,1), mode='evaluation', verbose=False)
mlops.inference(slice(0,10,1), mode='visualization', verbose=False)
mlops.inference(slice(0,10,1), mode='prediction', verbose=False)

mlops.training_board(log='commit')
mlops.drawup_source('20211221_204726-my_code.py')
mlops.display_source('20211221_204726-my_code.py')
```

```python
X = mlops.dataset.loc[:, mlops.dataset.columns != 'target']
y = mlops.dataset.loc[:, 'target']
model = mlops.model

pred_val = mlops.entry_point.predict(model, X)
metric = mlops.entry_point.evaluate(y, pred_val)
report = mlops.entry_point.report(metric)
```



### training_board
```python
from ailever.mlops import project
        
mlops = project({
    'root':'my_proeject',
    'feature_store':'my_fs', 
    'model_registry':'my_mr', 
    'source_repository':'my_sr', 
    'metadata_store':'my_ms'})

mlops.training_board(log='inside')
mlops.training_board(log='outside')
mlops.training_board(log='commit')
```
```python
from ailever.mlops import project
        
mlops = project({
    'root':'my_proeject',
    'feature_store':'my_fs', 
    'model_registry':'my_mr', 
    'source_repository':'my_sr', 
    'metadata_store':'my_ms'})

mlops.training_board(log='commit')
mlops.display_source('20211221_204726-my_code.py')
mlops.drawup_source('20211221_204726-my_code.py')
mlops.feature_choice('20211222_020514-dataset0.csv').model_choice('20211222_020514-ExtraTreesClassifier.joblib')

#mlops.prediction(slice(0,10,1)) 
#mlops.inference(slice(0,10,1), mode='evaluation', verbose=False)
#mlops.inference(slice(0,10,1), mode='visualization', verbose=False)
#mlops.inference(slice(0,10,1), mode='prediction', verbose=False)

pred_val = mlops.entry_point.predict(model, X)
metric = mlops.entry_point.evaluate(y, pred_val)
report = mlops.entry_point.report(metric)
```
