```python
from ailever.mlops import project
from ailever.dataset import SKAPI
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.linear_model import LogisticRegression
import xgboost

dataset = SKAPI.iris(download=False)

model0 = ExtraTreesClassifier()
model1 = LogisticRegression()
model2 = xgboost.XGBClassifier()

mlops = project({
    'root':'my_proeject',
    'feature_store':'my_fs', 
    'model_registry':'my_mr', 
    'source_repository':'my_sr', 
    'metadata_store':'my_ms'})

mlops.dataset = [dataset]
mlops.model = [model0, model1, model2]

# mlops.training_board()
mlops.inference(dataset.loc[10:30, dataset.columns!='target'])
```
```python
from ailever.mlops import project
from ailever.dataset import SKAPI
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.linear_model import LogisticRegression
import xgboost

dataset0 = SKAPI.iris(download=False)
dataset1 = SKAPI.digits(download=False)

model0 = ExtraTreesClassifier()
model1 = LogisticRegression()
model2 = xgboost.XGBClassifier()

mlops = project({
    'root':'my_proeject',
    'feature_store':'my_fs', 
    'model_registry':'my_mr', 
    'source_repository':'my_sr', 
    'metadata_store':'my_ms'})

mlops.dataset = [dataset0, dataset1]
mlops.model = [model0, model1, model2]

# mlops.training_board()
mlops.feature_choice(0).model_choice(2)
mlops.inference(dataset0.loc[10:30, dataset0.columns!='target'])
```
```python
# mlops.training_board()
mlops.feature_choice().model_choice()
mlops.inference(dataset.loc[:10, dataset.columns!='target'])
mlops.summary()
```
```python
mlops.training_board()
model = mlops.get_model()
dataset = mlops.get_dataset()
```


