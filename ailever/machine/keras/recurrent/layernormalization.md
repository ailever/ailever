## LayerNormalization
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization

```python
import numpy as np
import tensorflow as tf

# [Batch, Sequence, Dimension]
x = 10*np.arange(10).reshape(-1, 5, 2)         # x.shape: (1,5,2)
embedding = tf.constant(x, dtype=tf.float32)

layer = tf.keras.layers.LayerNormalization(axis=-1, epsilon=0.001)
norm1 = layer(embedding)

mean = tf.transpose(tf.math.reduce_mean(embedding, axis=-1), perm=[1,0], conjugate=None)[tf.newaxis, :, :]
var = tf.transpose(tf.math.reduce_variance(embedding, axis=-1), perm=[1,0], conjugate=None)[tf.newaxis, :, :]
norm2 = (embedding - mean)/tf.math.sqrt(var + layer.epsilon)
norm2 = tf.einsum('ijk,k->jk', norm2, layer.weights[0]) + layer.weights[1]
norm1 - norm2
```
