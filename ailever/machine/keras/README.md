## Neural Network Modeling
### Keras Review
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

model = models.Model(x, layer(x), name='CustomModel')
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
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer="random_normal", trainable=True)

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
    def __init__(self, name=None):
        super(CustomModel, self).__init__(name=name)
        self.dense = layers.Dense(5, activation='relu', name='CustomLayer')

    def call(self, x, training=None):
        x = self.dense(x)
        return x

x = tf.random.normal(shape=(32,2,8))
model = CustomModel(name='CustomModel')
model(x)

model.summary()
```

`Loss Class`
```python
import tensorflow as tf
from tensorflow.keras import losses

class CustomLoss(losses.Loss):
    def call(self, target, hypothesis, training=None):
        target = tf.cast(target, hypothesis.dtype)
        return tf.math.reduce_mean(tf.math.square(hypothesis - target), axis=-1)

target = tf.constant([[0., 1.], [0., 0.]])
hypothesis = tf.constant([[1., 1.], [1., 0.]])
cost = CustomLoss()
cost(target, hypothesis)
```

`LearningRateSchedule Class`
```python
import tensorflow as tf
from tensorflow.keras import optimizers

class CustomSchedule(optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, gradients):
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        return self.initial_learning_rate / (step + 1)


W = tf.Variable(4.0, trainable=True)
Y = tf.Variable(1.0, trainable=False)
with tf.GradientTape() as tape:
    cost = (W - Y)**2

# Caluation gradients
gradients = tape.gradient(cost, [W]); print('[gradient]:', gradients[0].numpy())

# Gradient scheduling 
lr_schedule = CustomSchedule(initial_learning_rate=0.1)

# Assgin gradient policy to trainable tensor
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
optimizer.apply_gradients(zip(gradients, [W])) # optimizer.weights[-1], optimizer.get_weights(), optimizer.set_weights(optimizer.weights)
W.numpy()
```


`Metric Class`
```python
import tensorflow as tf
from tensorflow.keras import metrics

class CustomMetric(metrics.Metric):
    def __init__(self, name='binary_true_positives', **kwargs):
        super(CustomMetric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        values = tf.cast(values, self.dtype)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, values.shape)
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

y_true = [0, 1, 1, 1]
y_pred = [1, 0, 1, 1]

metric = CustomMetric(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
```




---

## Reference


