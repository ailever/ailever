## Supervised Learning

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
mlops.feature_choice(0).model_choice(1)
mlops.inference(dataset0.loc[:10, dataset0.columns!='target'])
mlops.training_board() # mlops.training_board(log='inside')
```
```python
mlops.training_board() #mlops.insidelog
model = mlops.drawup_model('20211219_123402-LGBMClassifier.joblib')
dataset = mlops.drawup_dataset('20211219_123400-dataset0.csv')
```



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
mlops.inference(dataset.loc[:10, dataset.columns!='target'])
mlops.training_board(log='outside')
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
mlops.inference(slice(0,10,1))
mlops.training_board(log='inside')
```



