## [Deep Learning] | [tensorflow](https://www.tensorflow.org/api_docs/python/) | [github](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python)
- https://github.com/deeplearningzerotoall/TensorFlow/tree/master/tf_2.x

### Contents
- Installation
- Tensor
- Datasets
- Models
    - Linear Regression
    - Convolutional Neural Network
    - Recurrent Neural Network 

## Installation
```bash
$ pip install tensorflow
$ pip install tensorflow_datasets
$ pip install pydot
```

---

<br><br><br>

---

## Tensor
### Tensor Generation
`Tensor`
```python
import tensorflow as tf

tf.zeros(shape=(3,4), dtype=tf.float32)
tf.zeros_like(input=[1,2,3], dtype=tf.float32)
tf.ones(shape=(3,4), dtype=tf.float32)
tf.fill(dims=(3,4), value=5)
tf.constant(value=5, shape=(4,4))
tf.constant(value=[1,2,3], dtype=None, shape=None, name='Const')
tf.linspace(start=0, stop=1, num=100)
tf.range(start=0, limit=10, delta=1)

```

`Variable`
```python
import tensorflow as tf

tf.Variable([0.0])
```

<br><br><br>


---

## Datasets
- https://www.tensorflow.org/api_docs/python/tf/data/Dataset
- https://www.tensorflow.org/datasets/overview

```python
import tensorflow_datasets as tfds

tfds.list_builders()
```

### From dictionary
```python
import tensorflow as tf
from ailever.dataset import SKAPI

df = SKAPI.housing()
df = df.copy()

target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((dict(df), target)).shuffle(buffer_size=df.shape[0]).batch(5)

for features, targets in dataset.take(1):
    features.keys()
    print(features['MedInc'])
    print(features['HouseAge'])
    print(features['AveRooms'])
    print(features['AveBedrms'])
    print(features['Population'])
    print(features['AveOccup'])
    print(features['Latitude'])
    print(features['Longitude'])
    print(targets)
```
### From dataframe
```python
import tensorflow as tf
import tensorflow_datasets as tfds
from ailever.dataset import SKAPI

df = SKAPI.housing()
df = df.copy()

target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df, target)).shuffle(buffer_size=df.shape[0]).batch(5)

# tensor-return
for features, targets in dataset.take(1):
    print(features)
    print(target)

# ndarray-return    
for features, targets in tfds.as_numpy(dataset.take(1)):
    print(features)
    print(target)    
```


### Usage
```python
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

# [Data-Preprocessing Step]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channels dimension
x_train = x_train[..., tf.newaxis].astype("float32")
x_test = x_test[..., tf.newaxis].astype("float32")

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



# [Modeling Step]
class Architecture(Model):
    def __init__(self):
        super(Architecture, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation='relu')
        self.d2 = layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = Architecture(); print(model.summary())
criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



# [Training Step]
@tf.function
def train_step(features, targets):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(features, training=True)
        cost = criterion(targets, predictions)
    gradients = tape.gradient(cost, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(cost)
    train_accuracy(targets, predictions)
    
@tf.function
def test_step(features, targets):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(features, training=False)
    cost = criterion(targets, predictions)

    test_loss(cost)
    test_accuracy(targets, predictions)
    
EPOCHS = 5
for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for features, targets in train_dataset:
        train_step(features, targets)

    for features, targets in test_dataset:
        test_step(features, targets)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )

model.save_weights('model.ckpt')
model.load_weights('model.ckpt')
tf.keras.utils.plot_model(model, show_shapes=True)
```

<br><br><br>

---


## Models
### Linear Regression
`with gradient implementation`
```python

```
`with gradient tape`
```python
import tensorflow as tf
import numpy as np

# Data
x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

# W, b initialize
W = tf.Variable(2.9)
b = tf.Variable(0.5)

# W, b update
for i in range(100):
    # Gradient descent
    with tf.GradientTape() as tape:
        hypothesis = W * x_data + b
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

# predict
print(W * 5 + b)
print(W * 2.5 + b)
```

### Convolutnal Neural Network


### Recurrent Neural Network

<br><br><br>

---

## Tensorboard



