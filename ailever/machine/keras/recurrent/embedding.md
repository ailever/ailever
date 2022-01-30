## Embedding
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding

```python
import tensorflow as tf
from tensorflow.keras import layers

x = tf.random.shuffle(value=range(0,10), seed=None, name=None) # x: (10,)
layer = layers.Embedding(1000, 64, input_length=10)            # layer.weights[0]: (1000, 64) 
y = layer(x)                                                   # y: (10, 64)
```
