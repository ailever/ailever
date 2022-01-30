
## Dense Layer
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

### 2D Input
```python
import tensorflow as tf
from tensorflow.keras import layers

x = tf.random.normal(shape=(32, 1))
layer = layers.Dense(
    units=4, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None)
y = layer(x)
y_ = tf.einsum('ij,jk->ik', x, layer.weights[0]) + layer.weights[1]
y_ - y
```

### 3D Input
```python
import tensorflow as tf
from tensorflow.keras import layers

x = tf.random.normal(shape=(32, 1, 1))
layer = layers.Dense(
    units=4, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None)
y = layer(x)
y_ = tf.einsum('ijk,kl->ijl', x, layer.weights[0]) + layer.weights[1]  
y_ - y
```

### 4D Input
```python
import tensorflow as tf
from tensorflow.keras import layers

x = tf.random.normal(shape=(32, 1, 1, 1))
layer = layers.Dense(
    units=4, activation=None, use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None)
y = layer(x)
y_ = tf.einsum('ijkl,lm->ijkm', x, layer.weights[0]) + layer.weights[1]
y_ - y
```

