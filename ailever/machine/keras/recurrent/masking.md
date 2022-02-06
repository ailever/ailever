## Masking
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Masking

## Mask
### Keras Masking
```python
import tensorflow as tf
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

### Tensorflow Masking
```python
import tensorflow as tf

pad_sequences = tf.constant([
    [ 711,  632,   71,    0,    0,    0,],
    [  73,    8, 3215,   55,  927,    0,],
    [  83,   91,    1,  645, 1253,  927,]
])
tf.cast(pad_sequences, dtype=tf.bool)
```
```
<tf.Tensor: shape=(3, 6), dtype=bool, numpy=
array([[ True,  True,  True, False, False, False],
       [ True,  True,  True,  True,  True, False],
       [ True,  True,  True,  True,  True,  True]])>
```

### Numpy Masking
```python
import numpy as np
import tensorflow as tf

tf.constant(np.triu(np.full(fill_value=1, shape=(5, 8))), dtype=tf.bool)
```
```
<tf.Tensor: shape=(5, 8), dtype=bool, numpy=
array([[ True,  True,  True,  True,  True,  True,  True,  True],
       [False,  True,  True,  True,  True,  True,  True,  True],
       [False, False,  True,  True,  True,  True,  True,  True],
       [False, False, False,  True,  True,  True,  True,  True],
       [False, False, False, False,  True,  True,  True,  True]])>
```


```python
import numpy as np
import tensorflow as tf

tf.constant(np.tril(np.full(fill_value=1, shape=(5, 8))), dtype=tf.bool)
```
```
<tf.Tensor: shape=(5, 8), dtype=bool, numpy=
array([[ True, False, False, False, False, False, False, False],
       [ True,  True, False, False, False, False, False, False],
       [ True,  True,  True, False, False, False, False, False],
       [ True,  True,  True,  True, False, False, False, False],
       [ True,  True,  True,  True,  True, False, False, False]])>
```

```python
import numpy as np
import tensorflow as tf

mask = np.full(fill_value=1, shape=(5, 8))
mask[:, -2:].fill(0)
tf.constant(mask, dtype=tf.bool)
```
```
<tf.Tensor: shape=(5, 8), dtype=bool, numpy=
array([[ True,  True,  True,  True,  True,  True, False, False],
       [ True,  True,  True,  True,  True,  True, False, False],
       [ True,  True,  True,  True,  True,  True, False, False],
       [ True,  True,  True,  True,  True,  True, False, False],
       [ True,  True,  True,  True,  True,  True, False, False]])>
```


## Masked Embedding
- https://www.tensorflow.org/guide/keras/masking_and_padding
```python
import tensorflow as tf
from tensorflow.keras import layers

pad_sequences = tf.constant([
    [ 711,  632,   71,    0,    0,    0,],
    [  73,    8, 3215,   55,  927,    0,],
    [  83,   91,    1,  645, 1253,  927,]
])

layer = layers.Masking()
masked_embedding = layer(tf.cast(pad_sequences[:,:,tf.newaxis], tf.float32))
masked_embedding
```
```
<tf.Tensor: shape=(3, 6, 1), dtype=float32, numpy=
array([[[7.110e+02],
        [6.320e+02],
        [7.100e+01],
        [0.000e+00],
        [0.000e+00],
        [0.000e+00]],

       [[7.300e+01],
        [8.000e+00],
        [3.215e+03],
        [5.500e+01],
        [9.270e+02],
        [0.000e+00]],

       [[8.300e+01],
        [9.100e+01],
        [1.000e+00],
        [6.450e+02],
        [1.253e+03],
        [9.270e+02]]], dtype=float32)>
```
