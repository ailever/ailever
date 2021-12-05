```python
from ailever.mlops import project

mlops_bs = project({
    'root':'my_mlops',
    'feature_store':'my_fs', 
    'model_registry':'my_mr', 
    'source_repository':'my_sr', 
    'metadata_store':'my_ms'})
mlops_bs.core['MLOPS'].listdir()
```
