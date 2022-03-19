# Language Package

```python
from ailever.language import dashboard
dashboard()
```

## Dataset
```python
from nlp import list_datasets, load_dataset
import pandas as pd

datasets = load_dataset('imdb')
train_dataset = pd.DataFrame(datasets['train'])
test_dataset = pd.DataFrame(datasets['test'])
```
