```python
from ailever.mlops import project

mlops = project({
    'root':'my_proeject',
    'feature_store':'my_fs', 
    'model_registry':'my_mr', 
    'source_repository':'my_sr', 
    'metadata_store':'my_ms'})

mlops.dataset = dataset 
mlops.model = model
mlops.predict(dataset[:10])

mlops.feature_choice().model_choice()
mlops.predict(dataset[:10])
mlops.summary()
```

```python
model = mlops.get_model()
dataset = mlops.get_dataset()
```


