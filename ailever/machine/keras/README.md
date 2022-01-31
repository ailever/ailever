## Neural Network Modeling

`Sequential API`
```python
import tensorflow as tf
from tensorflow.keras import layers, models

x = tf.random.normal(shape=(32,2,8))
layer = layers.Dense(units=5, name='CustomLayer')

model = models.Sequential([layer], name='CustomModel')
model(x)

model.summary()
```

`Functional API`
```python
import tensorflow as tf
from tensorflow.keras import layers, models

x = layers.Input(shape=(2,8), name='CustomInput')
layer = layers.Dense(units=5, name='CustomLayer')

model = models.Model(x, layer(x))
model.summary()
```

`Layer Class`
```python
import tensorflow as tf
from tensorflow.keras import layers, models

class CustomLayer(layers.Layer):
    def __init__(self, units=32, name=None):
        super(CustomLayer, self).__init__(name=name)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, X, training=None):
        return tf.matmul(X, self.w) + self.b

    def get_config(self):
        return {"units": self.units}

x = tf.random.normal(shape=(32,2,8))
model = models.Sequential(CustomLayer(name='CustomLayer'), name='CustomModel')
model(x)

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

