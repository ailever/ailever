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

`Model Class`
```python
import tensorflow as tf
from tensorflow.keras import layers, models

class CustomModel(models.Model):
    def __init__(self, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.dense = layers.Dense(5, activation='relu', name='L1')

    def call(self, x):
        x = self.dense(x)
        return x

x = tf.random.normal(shape=(32,2,8))
model = CustomModel(name='CustomModel')
model(x)

model.summary()
```

