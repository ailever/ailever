## Neural Network Modeling

`Sequential API`
```python
import tensorflow as tf
from tensorflow.keras import layers, models

x = tf.random.normal(shape=(32,2,8))
layer = layers.Dense(units=5, name='L1')

model = models.Sequential()
model.add(layer)
model(x)

model.summary()
```

`Functional API`
```python
import tensorflow as tf
from tensorflow.keras import layers, models

x = layers.Input(shape=(2,8), name='Input')
layer = layers.Dense(units=5, name='L1')

model = models.Model(x, layer(x))
model.summary()
```

