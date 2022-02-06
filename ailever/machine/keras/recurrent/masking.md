## Masking
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking

```python
from tensorflow.keras import layers

pad_sequences = tf.constant([
    [ 711,  632,   71,    0,    0,    0,],
    [  73,    8, 3215,   55,  927,    0,],
    [  83,   91,    1,  645, 1253,  927,]
])

layer = layers.Masking()
masked_embedding = layer(tf.cast(pad_sequences[:,:,tf.newaxis], tf.float32))
masked_embedding._keras_mask
```
```
<tf.Tensor: shape=(3, 6), dtype=bool, numpy=
array([[ True,  True,  True, False, False, False],
       [ True,  True,  True,  True,  True, False],
       [ True,  True,  True,  True,  True,  True]])>
```




