## Installation
```bash
$ pip install tensorflow
$ pip install tensorflow_datasets
```

## Datasets
- https://www.tensorflow.org/api_docs/python/tf/data/Dataset
- https://www.tensorflow.org/datasets/overview
### From dictionary
```python
import tensorflow as tf
from ailever.dataset import SKAPI

df = SKAPI.housing()
df = df.copy()

target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((dict(df), target)).shuffle(buffer_size=df.shape[0]).batch(5)

for features, target in dataset.take(1):
    features.keys()
    print(features['MedInc'])
    print(features['HouseAge'])
    print(features['AveRooms'])
    print(features['AveBedrms'])
    print(features['Population'])
    print(features['AveOccup'])
    print(features['Latitude'])
    print(features['Longitude'])
    print(target)
```
### From dataframe
```python
import tensorflow as tf
from ailever.dataset import SKAPI

df = SKAPI.housing()
df = df.copy()

target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df, target)).shuffle(buffer_size=df.shape[0]).batch(5)

for features, target in dataset.take(1):
    features.keys()
    print(features['MedInc'])
    print(features['HouseAge'])
    print(features['AveRooms'])
    print(features['AveBedrms'])
    print(features['Population'])
    print(features['AveOccup'])
    print(features['Latitude'])
    print(features['Longitude'])
    print(target)
```


