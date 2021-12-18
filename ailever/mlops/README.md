```python
from ailever.mlops import project
from ailever.dataset import SKAPI
from sklearn import ensemble

dataset = SKAPI.iris(download=False)
model = ensemble.ExtraTreesClassifier()

mlops = project({
    'root':'my_proeject',
    'feature_store':'my_fs', 
    'model_registry':'my_mr', 
    'source_repository':'my_sr', 
    'metadata_store':'my_ms'})

mlops.dataset = dataset 
mlops.model = model
mlops.inference(dataset.loc[:10, dataset.columns!='target'])
```
```
mlops.training_board()
mlops.feature_choice().model_choice()
mlops.inference(dataset.loc[:10, dataset.columns!='target'])
mlops.summary()
```
```python
model = mlops.get_model()
dataset = mlops.get_dataset()
```


