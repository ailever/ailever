## [Deep Learning] | [tensorflow](https://www.tensorflow.org/api_docs/python/) | [github](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python)
- https://www.tensorflow.org/guide
- https://www.tensorflow.org/tutorials
- https://github.com/deeplearningzerotoall/TensorFlow/tree/master/tf_2.x
    
### Contents
- https://www.tensorflow.org/api_docs/python/tf/dtypes
- https://www.tensorflow.org/api_docs/python/tf/math
- https://www.tensorflow.org/api_docs/python/tf/linalg
- https://www.tensorflow.org/api_docs/python/tf/signal
- https://www.tensorflow.org/api_docs/python/tf/nn
- https://www.tensorflow.org/api_docs/python/tf/keras
- https://www.tensorflow.org/api_docs/python/tf/initializers
- https://www.tensorflow.org/api_docs/python/tf/losses
- https://www.tensorflow.org/api_docs/python/tf/optimizers
- https://www.tensorflow.org/api_docs/python/tf/metrics

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

### Eager Execution
```python
import tensorflow as tf

tf.executing_eagerly()                        # True
tf.convert_to_tensor([1,2,3])

# tensorflow.python.framework.ops.EagerTensor
isinstance(tf.constant([1, 2, 3]), tf.Tensor)   # True
isinstance(tf.constant([1, 2, 3]), tf.Variable) # False

# tensorflow.python.ops.resource_variable_ops.ResourceVariable
isinstance(tf.Variable([1, 2, 3]), tf.Tensor)   # False
isinstance(tf.Variable([1, 2, 3]), tf.Variable) # True
```

`Device Dependency`
```python
import tensorflow as tf
import time

def measure(x, steps):
    # device initializer
    tf.matmul(x, x)
    
    start = time.time()
    for i in range(steps):
        x = tf.matmul(x, x)

    _ = x.numpy()
    end = time.time()
    return end - start

# on CPU:
with tf.device("/cpu:0"):
    print("CPU: {} 초".format(measure(tf.random.normal(shape=(1000, 1000)), steps=200)))

# on GPU:
if tf.config.experimental.list_physical_devices("GPU"):
    with tf.device("/gpu:0"):
        print("GPU: {} 초".format(measure(tf.random.normal(shape=(1000, 1000)), steps=200)))
else:
    print("GPU: 없음")
```
```python
import tensorflow as tf

x = tf.random.normal([10, 10])
if tf.config.experimental.list_physical_devices("GPU"):
    x_gpu = x.gpu()
    x_cpu = x.cpu()
    
    _ = tf.matmul(x_cpu, x_cpu)    # CPU에서 실행
    _ = tf.matmul(x_gpu0, x_gpu0)  # GPU:0에서 실행
```

### Tensor Datatype
```python
import tensorflow as tf

tf.constant(value=True, dtype=tf.dtypes.bool)
tf.constant(value=2**7-1, dtype=tf.dtypes.int8)
tf.constant(value=2**15-1, dtype=tf.dtypes.int16)
tf.constant(value=2**31-1, dtype=tf.dtypes.int32)
tf.constant(value=2**63-1, dtype=tf.dtypes.int64)
tf.constant(value=2**8-1, dtype=tf.dtypes.uint8)
tf.constant(value=2**16-1, dtype=tf.dtypes.uint16)
tf.constant(value=2**32-1, dtype=tf.dtypes.uint32)
tf.constant(value=2**64-1, dtype=tf.dtypes.uint64)
tf.constant(value=1., dtype=tf.dtypes.float16) # 1/5/10
tf.constant(value=1., dtype=tf.dtypes.float32) # 1/8/22
tf.constant(value=1., dtype=tf.dtypes.float64) # 1/11/52
tf.constant(value=1., dtype=tf.dtypes.double)
tf.constant(value='string', dtype=tf.dtypes.string)
tf.constant(1+2j, dtype=tf.dtypes.complex64)
tf.constant(1+2j, dtype=tf.dtypes.complex128)

x = tf.constant(value=1., dtype=tf.dtypes.float64)
tf.cast(x, dtype=tf.dtypes.bool)
```
### Tensor Generation
`Tensor`
```python
import tensorflow as tf

# tensor
tf.zeros(shape=(3,4), dtype=tf.dtypes.float32)
tf.zeros_like(input=[1,2,3], dtype=tf.dtypes.float32)
tf.ones(shape=(3,4), dtype=tf.dtypes.float32)
tf.ones_like(input=[1,2,3], dtype=tf.dtypes.float32)
tf.fill(dims=(3,4), value=5)
tf.eye(num_rows=10, num_columns=None, batch_shape=None, dtype=tf.dtypes.float32, name=None)
tf.constant(value=5, shape=(4,4))
tf.constant(value=[1,2,3], dtype=None, shape=None, name='Const')

# sequence
tf.linspace(start=0, stop=1, num=100)
tf.range(start=0, limit=10, delta=1)

# probability distribution
tf.random.shuffle(value=range(0,10), seed=None, name=None)
tf.random.categorical(logits=tf.math.log([[0.5, 0.5]]), num_samples=10, dtype=None, seed=None, name=None)
tf.random.poisson(shape=(10,), lam=[0.5, 1.5, 3.], dtype=tf.dtypes.float32, seed=None, name=None)
tf.random.uniform(shape=(4,3), minval=0, maxval=None, dtype=tf.dtypes.float32, seed=None, name=None)
tf.random.normal(shape=(5,5), mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
tf.random.gamma(shape=(4,3), alpha=0.1, beta=None, dtype=tf.dtypes.float32, seed=None, name=None)
tf.random.stateless_binomial(shape=(3,2), counts=[10., 20.], probs=0.5, output_dtype=tf.dtypes.int32, name=None, seed=[123,456])
tf.random.stateless_categorical(logits=tf.math.log([[0.5, 0.5]]), num_samples=5, seed=[7, 17])
tf.random.stateless_gamma(shape=(10, 2), seed=[12, 34], alpha=[0.5, 1.5])
tf.random.stateless_normal(shape=(10, 2), seed=[12, 34], mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, name=None,alg='auto_select')
tf.random.stateless_uniform(shape=(10, 2), seed=[12, 34], minval=0, maxval=None, dtype=tf.dtypes.float32, name=None,alg='auto_select')
```

`Variable`
```python
import tensorflow as tf

tf.Variable([0.0])
```

### Tensor Manipulation
```python
import tensorflow as tf

# squeeze
tf.squeeze([[[1,2,3]]], axis=None)

# expand_dims
tf.expand_dims([1,2,3], axis=1)
tf.constant([1,2,3])[:, tf.newaxis]

# stack
x = tf.constant([1, 4])
y = tf.constant([2, 5])
z = tf.constant([3, 6])
tf.stack([x,y,z], axis=0)

# unstack
tf.unstack([[1, 2, 3], [4, 5, 6]], axis=0)

# concat
x = [[1, 2, 3],
     [4, 5, 6]]
y = [[7 , 8 , 9 ],
     [10, 11, 12]]
tf.concat([x, y], axis=0)

# split
tf.split([1,2,3,4], num_or_size_splits=2, axis=-1)

# repeat
tf.repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=0)

# tile
tf.tile([[["Repeat"]]], [1, 3, 1])

# unique
tf.unique([1,1,2,2,3,3], out_idx=tf.dtypes.int32, name=None)

# reshape
tf.reshape(shape=(3,2),
    tensor=[[1,2,3], 
            [4,5,6]])

# transpose
tf.transpose([[1, 2, 3], [4, 5, 6]], perm=[1,0], conjugate=None)

# rank
tf.rank([[1, 2], [3, 4]])

# shape
tf.shape([[1, 2], [3, 4]])

# size
tf.size([[1, 2], [3, 4]])

# sort
tf.sort([5,4,3,2,1], axis=-1, direction='ASCENDING', name=None)

# where
condition = [True, False, False, True]; yes = 1; no = 0
tf.where(condition, 1, 0)
```


### Tensor Mathmatics
```python
import tensorflow as tf

c = tf.constant(dtype=tf.dtypes.float32,
    value=[[1,-1,3], 
           [3,7,9]])
tf.math.argmax(c, axis=1, output_type=tf.dtypes.int64, name=None)
tf.math.argmin(c, axis=1, output_type=tf.dtypes.int64, name=None)
tf.math.reduce_max(c, axis=1, keepdims=False, name=None)
tf.math.reduce_min(c, axis=1, keepdims=False, name=None)
tf.math.reduce_sum(c, axis=1, keepdims=False, name=None)
tf.math.reduce_prod(c, axis=1, keepdims=False, name=None)
tf.math.reduce_mean(c, axis=1, keepdims=False, name=None)
tf.math.reduce_variance(c, axis=1, keepdims=False, name=None)
tf.math.reduce_std(c, axis=1, keepdims=False, name=None)

c1 = tf.constant([1,1,1])
c2 = tf.constant([2,3,4])
tf.math.add(c1, c2, name=None)
tf.math.add_n([c1, c2, c1], name=None)
tf.math.subtract(c1, c2, name=None)
tf.math.divide(c1, c2, name=None)
tf.math.multiply(c1, c2, name=None)
tf.math.minimum(c1, c2)
tf.math.maximum(c1, c2)

c = tf.constant(dtype=tf.dtypes.float32,
    value=[[.1,.2,.3],
           [.4,.5,.6]])
tf.identity(c, name=None)
tf.math.abs(c, name=None)
tf.math.sin(c, name=None)
tf.math.cos(c, name=None)
tf.math.tan(c, name=None)
tf.math.asin(c, name=None)
tf.math.acos(c, name=None)
tf.math.atan(c, name=None)
tf.math.asinh(c, name=None)
tf.math.acosh(c, name=None)
tf.math.atanh(c, name=None)
tf.math.exp(c, name=None)
tf.math.log(c, name=None)
tf.math.pow(c, 2, name=None)
tf.math.sqrt(c)
tf.math.square(c)
tf.math.sigmoid(c)

c = tf.constant(1+3j, dtype=tf.dtypes.complex64)
tf.math.real(c)
tf.math.imag(c)

# einsum notation
m0 = tf.random.normal(shape=[2, 3])
m1 = tf.random.normal(shape=[3, 5])
tf.einsum('ij,jk->ik', m0, m1)
```

### Linear Algebra
```python
import tensorflow as tf

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
tf.linalg.matmul(a, b)
```


### Variable Gradient
```python
import tensorflow as tf

w1 = tf.Variable(2.0, trainable=True, name='W1Tensor')  # w1.trainable = True
w2 = tf.Variable(2.0, trainable=False, name='W2Tensor') # w2.trainable = False

with tf.GradientTape() as tape:
    out1 = w1 * w1
    out2 = w2 * w2
    cost = out1 + out2

gradients = tape.gradient(cost, {'w1': w1, 'w2': w2})

print('[d(cost)/d(w1)]:', gradients['w1'])  # 2*x => 4
print('[d(cost)/d(w2)]:', gradients['w2'])  # None
w1.assign_sub(0.01*gradients['w1'])
w2.assign_sub(0.01*4)
```
```python
import tensorflow as tf

w1 = tf.Variable(2.0, trainable=True, name='W1Tensor')
w2 = tf.Variable(2.0, trainable=False, name='W2Tensor')

with tf.GradientTape() as tape:
    out1 = w1 * w1
    out2 = w2 * w2
    cost = out1 + out2

gradients = tape.gradient(cost, [w1, w2])
print('[d(cost)/d(w1)]:', gradients[0])  # 2*(w1) => 4
print('[d(cost)/d(w2)]:', gradients[1])  # None
w1.assign_sub(0.01*gradients[0])
w2.assign_sub(0.01*4)
```
`tf.function`
```python
import tensorflow as tf

@tf.function
def forward(x, y):
    return (x-y)**2

W = tf.Variable(2.0)
with tf.GradientTape() as tape:
    result = forward(W, 1.0)
tape.gradient(result, W)
```


<br><br><br>


---

## Module
- https://www.tensorflow.org/guide/intro_to_modules

### Module Class
```python
import tensorflow as tf

class Arcitecture(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.W = tf.Variable(4.0, trainable=True, name="trainable")
        self.Y = tf.Variable(2.0, trainable=False, name="non-trainable")
    
    def __call__(self, x):
        return tf.reduce_mean((self.W * x - self.Y)**2)

model = Arcitecture()
with tf.GradientTape() as tape:
    cost = model(tf.constant([1.,2.,3.]))
gradients = tape.gradient(cost, model.trainable_variables) # <-> model.variables [all variables]
model.trainable_variables[0].assign_sub(0.01*gradients[0])
model.trainable_variables[0]

#model.variables
#model.trainable_variables
#model.non_trainable_variables

#model.summary()
#tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
```

### Layer
`Custom Layer`
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

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

# Sequential Model
model = models.Sequential(name='CustomModel')
model.add(CustomLayer(10, name='CustomLayer'))
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,10)))

# Functional Model
inputs = layers.Input((100,))
outputs = CustomLayer(10)(inputs)
model = models.Model(inputs, outputs, name='CustomModel')
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,10)))

# prediction
model(tf.random.normal(shape=(1,100)))

# model entities
model.submodules
model.submodules[-1].input
model.submodules[-1].output
model.layers
model.variables
model.trainable_variables
model.non_trainable_variables

model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
```

`Layers through sequential`
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

sequential_model = models.Sequential()
sequential_model.add(layers.Dense(4, name='1L', activation="relu"))
sequential_model.add(layers.Dense(4, name='2L', activation="relu"))
sequential_model.add(layers.Dense(4, name='3L'))

y = sequential_model(tf.ones((1, 100)))
sequential_model.layers #sequential_model.submodules
sequential_model.get_layer(name='1L') # sequential_model.inputs
sequential_model.get_layer(name='2L')
sequential_model.get_layer(name='3L') # sequential_model.outputs

sequential_model.layers[0].trainable = False
sequential_model.layers[1].trainable = False
sequential_model.layers[2].trainable = True

sequential_model.variables
sequential_model.trainable_variables
sequential_model.non_trainable_variables

sequential_model.layers[-1].weights
sequential_model.layers[-1].input
sequential_model.layers[-1].output

sequential_model.summary()
tf.keras.utils.plot_model(sequential_model, "model.png", show_shapes=True)
```

### Model
`Layer` vs `Model`
- model.fit(), model.evaluate(), model.predict()
- save and serialization API(save(), save_weights()...)

`Sequential API`
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

sequential_model = models.Sequential()
sequential_model.add(layers.Dense(4, name='1L', activation="relu"))
sequential_model.add(layers.Dense(4, name='2L', activation="relu"))
sequential_model.add(layers.Dense(4, name='3L'))
#sequential_model(tf.ones((1, 100)))
#model = models.Model(sequential_model.inputs, sequential_model.outputs)
model = sequential_model

# training
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,4)))

# prediction
model(tf.random.normal(shape=(1,100)))

# save & load
model.save('model.h5')  # creates a HDF5 file 'my_model.h5'
model = models.load_model('model.h5')

# model entities
model.submodules
model.submodules[-1].input
model.submodules[-1].output
model.layers
model.variables
model.trainable_variables
model.non_trainable_variables

model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
```

`Functional API`
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

# layers
inputs = layers.Input(shape=(None, None, 10))
processed = layers.RandomCrop(width=32, height=32)(inputs)
conv = layers.Conv2D(filters=2, kernel_size=3)(processed)
pooling = layers.GlobalAveragePooling2D()(conv)
outputs = layers.Dense(10)(pooling)

# models
backbone = models.Model(processed, conv)
activations = models.Model(conv, outputs)
model = models.Model(inputs, outputs)

# training
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.fit(tf.random.normal(shape=(100, 1, 1, 10)), tf.random.normal(shape=(100, 10)))

# prediction
model(tf.random.normal(shape=(1, 1, 1, 10)))

# save & load
model.save('model.h5')  # creates a HDF5 file 'my_model.h5'
model = models.load_model('model.h5')

# model entities
model.submodules
model.submodules[-1].input
model.submodules[-1].output
model.layers
model.variables
model.trainable_variables
model.non_trainable_variables

model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
```

`Model Class`
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

class CustomModel(models.Model):
    def __init__(self, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.dense_1 = layers.Dense(64, activation='relu', name='L1')
        self.dense_2 = layers.Dense(10, name='L2')

    def call(self, inputs):
        x = self.dense_1(inputs)
        return self.dense_2(x)

# training
model = CustomModel(name='CustomModel')
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,10)))

# prediction
model(tf.random.normal(shape=(1,100)))

# model entities
model.layers
model.variables
model.trainable_variables
model.non_trainable_variables

model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
```

### Applications
```python
import tensorflow as tf

""" [Supported models]
'DenseNet121', 'DenseNet169', 'DenseNet201',
'EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 
'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7',
'InceptionResNetV2', 'InceptionV3',
'MobileNet', 'MobileNetV2', 'MobileNetV3Large', 'MobileNetV3Small',
'NASNetLarge', 'NASNetMobile',
'ResNet101', 'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50', 'ResNet50V2',
'VGG16', 'VGG19', 'Xception']
"""

model = tf.keras.applications.VGG19()
model.submodules
model.layers
model.variables
model.trainable_variables
model.non_trainable_variables

model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
```


---


## Datasets
- https://www.tensorflow.org/api_docs/python/tf/data/Dataset
- https://www.tensorflow.org/datasets/overview
- https://www.tensorflow.org/guide/data
- https://www.tensorflow.org/guide/data_performance
- https://www.tensorflow.org/guide/data_performance_analysis

### Built-in Dataset
`keras.datasets`
```python
import tensorflow as tf

#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data()
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```
`tensorflow_datasets`
```python
import tensorflow_datasets as tfds

tfds.list_builders()
```

### Tensorflow Dataset
- https://www.tensorflow.org/api_docs/python/tf/data/Dataset

`Extraction: tf.data.Dataset.flat_map`
```python
# tf.data.Dataset.interleave() is a generalization of flat_map, since flat_map produces the same output as tf.data.Dataset.interleave(cycle_length=1)
import tensorflow as tf

dataset = [[1, 2, 3], 
           [4, 5, 6], 
           [7, 8, 9]]
iterable_dataset = tf.data.Dataset.from_tensor_slices(dataset)
iterable_dataset = iterable_dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)) 
list(iterable_dataset.as_numpy_iterator())
```

`Extraction: tf.data.Dataset.interleave`
```python
# The cycle_length and block_length arguments control the order in which elements are produced.
# The cycle_length controls the number of input elements that are processed concurrently.
# If you set cycle_length to 1, this transformation will handle one input element at a time, and will produce identical results to tf.data.Dataset.flat_map.
# In general, this transformation will apply map_func to cycle_length input elements, 
# open iterators on the returned Dataset objects, 
# and cycle through them producing block_length consecutive elements from each iterator,
# and consuming the next input element each time it reaches the end of an iterator.

# NOTE: New lines indicate "block" boundaries.
from collections import Counter
import pandas as pd
import tensorflow as tf

dataset = tf.data.Dataset.range(1, 11)  # ==> [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
dataset = dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(20), cycle_length=4, block_length=7)

display(pd.DataFrame({ k:[v] for k,v in Counter(dataset.as_numpy_iterator()).items()}).T.rename(columns={0:'CNT'}))
print('TOTALNUM:', len(list(dataset.as_numpy_iterator())))
print('ELEMENTS:', list(dataset.as_numpy_iterator()))
```
```
[1, 1, 1, 1, 1, 1, 1, 
 2, 2, 2, 2, 2, 2, 2, 
 3, 3, 3, 3, 3, 3, 3, 
 4, 4, 4, 4, 4, 4, 4,
 
 1, 1, 1, 1, 1, 1, 1, 
 2, 2, 2, 2, 2, 2, 2, 
 3, 3, 3, 3, 3, 3, 3, 
 4, 4, 4, 4, 4, 4, 4, 
 
 1, 1, 1, 1, 1, 1, 
 2, 2, 2, 2, 2, 2, 
 3, 3, 3, 3, 3, 3, 
 4, 4, 4, 4, 4, 4, 
 
 5, 5, 5, 5, 5, 5, 5, 
 6, 6, 6, 6, 6, 6, 6, 
 7, 7, 7, 7, 7, 7, 7, 
 8, 8, 8, 8, 8, 8, 8, 
 
 5, 5, 5, 5, 5, 5, 5, 
 6, 6, 6, 6, 6, 6, 6, 
 7, 7, 7, 7, 7, 7, 7, 
 8, 8, 8, 8, 8, 8, 8,
 
 5, 5, 5, 5, 5, 5, 
 6, 6, 6, 6, 6, 6, 
 7, 7, 7, 7, 7, 7, 
 8, 8, 8, 8, 8, 8,
 
  9,  9,  9,  9,  9,  9,  9, 
 10, 10, 10, 10, 10, 10, 10, 
  9,  9,  9,  9,  9,  9,  9, 
 10, 10, 10, 10, 10, 10, 10,
 
  9,  9,  9,  9,  9,  9, 
 10, 10, 10, 10, 10, 10]
```

`Preprocessing: tf.data.Dataset.map`
```python
import tensorflow as tf

iterable_dataset = tf.data.Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
iterable_dataset = iterable_dataset.map(lambda x: x + 1)
list(iterable_dataset.as_numpy_iterator())
```
```python
import tensorflow as tf

def preprocessing(x):
    return x+1

iterable_dataset = tf.data.Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
iterable_dataset = iterable_dataset.map(preprocessing)
list(iterable_dataset.as_numpy_iterator())
```


### Tensorflow Iterable Dataset
- https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
- https://cyc1am3n.github.io/2018/09/13/how-to-use-dataset-in-tensorflow.html
- http://learnwebgl.brown37.net/rendering/interleaved_buffers.html
- Ex> One-shot Iterator, Initializable iterator, Reinitializable Iterator, Feedable Iterator


`tf.data.Dataset.range`
```python
import tensorflow as tf

iterable_dataset = tf.data.Dataset.range(2)
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

size = 2
iterable_dataset = tf.data.Dataset.range(size).repeat(count=2)
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

size = 3
iterable_dataset = tf.data.Dataset.range(size).shuffle(buffer_size=size)
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

size = 5
iterable_dataset = tf.data.Dataset.range(size).batch(batch_size=2, drop_remainder=False)
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

size = 6
iterable_dataset = tf.data.Dataset.range(size).shuffle(buffer_size=size).batch(batch_size=2)
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

size = 6
iterable_dataset = tf.data.Dataset.range(size).shuffle(buffer_size=size).batch(batch_size=2)
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

size = 6
iterable_dataset = tf.data.Dataset.range(size).repeat(count=3).shuffle(buffer_size=size).batch(batch_size=2)
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

size = 6
iterable_dataset = tf.data.Dataset.range(size).repeat(count=3).shuffle(buffer_size=size).batch(batch_size=2).take(5)
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
iterable_dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))

for batch in iterable_dataset.batch(4).take(4):
    print([arr.numpy() for arr in batch])
```


`tf.data.Dataset.from_tensors`
```python
import tensorflow as tf

iterable_dataset = tf.data.Dataset.from_tensors(tf.constant(100))
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
```
```python
import tensorflow as tf

iterable_dataset = tf.data.Dataset.from_tensors(tf.constant(100)).repeat(2)
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
```

`tf.data.Dataset.from_tensor_slices`
```python
import tensorflow as tf

dataset_ = tf.random.normal(shape=(3,5))
iterable_dataset = tf.data.Dataset.from_tensor_slices((dataset_, ))

list(iterable_dataset.as_numpy_iterator())
list(iterable_dataset)
```
```python
import tensorflow as tf

dataset_ = tf.random.normal(shape=(3,5))
iterable_dataset = tf.data.Dataset.from_tensor_slices((dataset_, ))
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

iterable_dataset = tf.data.Dataset.from_tensor_slices(['a', 'b', 'c'])
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```


`tf.data.Dataset.from_generator`
```python
import tensorflow as tf

def generator(stop):
    element = 0
    while element < stop:
        yield element
        element += 1
    
iterable_dataset = tf.data.Dataset.from_generator(generator, args=[5], output_types=tf.int32, output_shapes = (), )
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

def generator(stop):
    iterator = range(stop)
    yield from iterator
    
iterable_dataset = tf.data.Dataset.from_generator(generator, args=[5], output_types=tf.int32, output_shapes = (), )
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

class CustomDataset(tf.data.Dataset):
    def _generator(stop):
        for element in range(stop):
            yield (element,)

    def __new__(cls, stop):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(stop,),
            output_types=tf.dtypes.int64,
            output_shapes=(1,))
    
iterable_dataset = CustomDataset(stop=5)
dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```

```python
import tensorflow as tf
    
class CustomDataset(tf.data.Dataset):
    def _generator(stop):
        for element in range(stop):
            yield (element,)

    def __new__(cls, stop):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(stop,),
            output_types=tf.dtypes.int64,
            output_shapes=(1,))

iterable_dataset = tf.data.Dataset.range(10).interleave(lambda x: CustomDataset(stop=5), cycle_length=1)
dataset_iterator = iter(iterable_dataset)
for element in dataset_iterator:
    print(element)
```



### Data Pipeline

#### Class Structure for Data-Pipelining
```python
import itertools
from collections import defaultdict

class CustomDataset:
    _BATCH_COUNTER = itertools.count()  # 생성된 데이터셋 수
    _EPOCHS_COUNTER = defaultdict(itertools.count)  # 각 데이터를 수행한 에포크 수
    def __new__(cls):
        return next(cls._BATCH_COUNTER)

next(CustomDataset._BATCH_COUNTER)
next(CustomDataset._EPOCHS_COUNTER[0])
```
```python
CustomDataset()
print('%-20s'%'_EPOCHS_COUNTER', CustomDataset._EPOCHS_COUNTER[0])
print('%-20s'%'_BATCH_COUNTER', CustomDataset._BATCH_COUNTER)
```


#### Data Extraction
`tf.data.Dataset.interleave`
```python
import tensorflow as tf

dataset = [[1, 2, 3], 
           [4, 5, 6], 
           [7, 8, 9]]
iterable_dataset = tf.data.Dataset.from_tensor_slices(dataset)
iterable_dataset = iterable_dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(10), cycle_length=4, block_length=7)
list(iterable_dataset)
```

```python
import tensorflow as tf
import pandas as pd
from collections import Counter

dataset = [[0], [1], [2], [3], [4]]
iterable_dataset = tf.data.Dataset.from_tensor_slices(dataset)
iterable_dataset = iterable_dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(10), cycle_length=4, block_length=7)
display(pd.DataFrame({k:[v] for k,v in Counter(list(map(lambda x: x.item(), iterable_dataset.as_numpy_iterator()))).items()}).T.rename(columns={0:'CNT'}))
```

`Dataset Class Structuralization(1)`
```python
import tensorflow as tf
from collections import Counter
import pandas as pd

class CustomDataset(tf.data.Dataset):
    def _generator(stop):
        for element in range(stop):
            yield (element,)

    def __new__(cls, stop):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(stop,),
            output_types=tf.dtypes.int64,
            output_shapes=(1,))

BATCH_SIZE = 5
iterable_dataset = tf.data.Dataset.range(10).interleave(lambda x: CustomDataset(BATCH_SIZE), cycle_length=1)
display(pd.DataFrame({k:[v] for k,v in Counter(list(map(lambda x: x.item(), iterable_dataset.as_numpy_iterator()))).items()}).T.rename(columns={0:'CNT'}))
```

`Dataset Class Structuralization(2)`
```python
import tensorflow as tf
from collections import Counter
import pandas as pd

class CustomDataset(tf.data.Dataset):
    def _generator(stop):
        for element in range(stop):
            yield (element,)

    def __new__(cls, stop):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(stop,),
            output_types=tf.dtypes.int64,
            output_shapes=(1,))

BATCH_SIZE = 5
def Extraction(*args, **kwargs):    
    tf.print('Data Extraction')
    return CustomDataset(BATCH_SIZE)

iterable_dataset = tf.data.Dataset.range(10).interleave(Extraction, cycle_length=1)
display(pd.DataFrame({k:[v] for k,v in Counter(list(map(lambda x: x.item(), iterable_dataset.as_numpy_iterator()))).items()}).T.rename(columns={0:'CNT'}))
```

`Dataset Class Structuralization(3)`
```python
import itertools
from collections import defaultdict, Counter
import pandas as pd
import tensorflow as tf

class CustomDataset(tf.data.Dataset):
    _BATCH_COUNTER = itertools.count()
    _EPOCHS_COUNTER = defaultdict(itertools.count)
    
    def _generator(batch_idx, batch_size):
        epoch_idx = next(CustomDataset._EPOCHS_COUNTER[batch_idx])                        
        for sample_idx in range(batch_size):
            yield (sample_idx,)

    def __new__(cls, batch_size):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(next(cls._BATCH_COUNTER), batch_size),
            output_types=tf.dtypes.int64,
            output_shapes=(1,))

BATCH_SIZE = 5
def Extraction(*args, **kwargs):    
    tf.print('Data Extraction')
    return CustomDataset(BATCH_SIZE)

iterable_dataset = tf.data.Dataset.range(10).interleave(Extraction, cycle_length=1)
display(pd.DataFrame({k:[v] for k,v in Counter(list(map(lambda x: x.item(), iterable_dataset.as_numpy_iterator()))).items()}).T.rename(columns={0:'CNT'}))
```

`Dataset Class Structuralization(4)`
```python
import itertools
from collections import defaultdict
import pandas as pd
import tensorflow as tf

class CustomDataset(tf.data.Dataset):
    _INSTANCE_COUNTER = itertools.count()
    _EPOCHS_COUNTER = defaultdict(itertools.count)
    
    def _generator(instance_idx, batch_size):
        epoch_idx = next(CustomDataset._EPOCHS_COUNTER[instance_idx])
        for sample_idx in range(batch_size):
            yield (instance_idx, epoch_idx, sample_idx, )

    def __new__(cls, batch_size):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(next(cls._INSTANCE_COUNTER), batch_size),
            output_types=tf.dtypes.int64,
            output_shapes=(3,))

BATCH_SIZE = 5
def Extraction(*args, **kwargs):    
    tf.print('Data Extraction')
    return CustomDataset(BATCH_SIZE)

def IterableDataset(*args):
    return tf.data.Dataset.range(10).interleave(Extraction, cycle_length=1)

display(pd.DataFrame(data=IterableDataset().as_numpy_iterator(), columns=['instance_idx', 'epoch_idx', 'sample_idx']).set_index(['instance_idx', 'epoch_idx', 'sample_idx']))
```

`Dataset Class Structuralization(5)`
```python
import itertools
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import tensorflow as tf

# Define Dataset
dataset = tf.random.normal(shape=(100, 7)).numpy()
dataset = pd.DataFrame(dataset).add_prefix('COMP')

# Parameters for Parallelized Data Extraction
NUM_ROWS = dataset.shape[0]
BATCH_SIZE = 5
EPOCHS = 2

class CustomDataset(tf.data.Dataset):
    _BATCH_COUNTER = itertools.count()
    _EPOCHS_COUNTER = defaultdict(itertools.count)
    # OUTPUT: (indices, values)    
    OUTPUT_TYPES = (tf.dtypes.int32, tf.dtypes.float32)
    OUTPUT_SHAPES = ((4, ), (1, 7))
    
    def _generator(batch_idx, batch_size):
        epoch_idx = next(CustomDataset._EPOCHS_COUNTER[batch_idx])
        for sample_idx, (row_idx, row_series) in enumerate(dataset.iloc[batch_idx*batch_size:(batch_idx+1)*batch_size].iterrows()):
            yield ([batch_idx, epoch_idx, sample_idx, row_idx], [row_series.values])

    def __new__(cls, batch_size):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(next(cls._BATCH_COUNTER), batch_size),
            output_types=cls.OUTPUT_TYPES,
            output_shapes=cls.OUTPUT_SHAPES)

def Extraction(*args, **kwargs):
    tf.print("Data Extraction")
    return CustomDataset(BATCH_SIZE)

def IterableDataset(num_repeat=1):
    return tf.data.Dataset.range(num_repeat).interleave(Extraction, cycle_length=1).batch(BATCH_SIZE, drop_remainder=True)

for batch_idx in range(NUM_ROWS//BATCH_SIZE):
    for indices, features in IterableDataset(num_repeat=EPOCHS):
        indices = pd.DataFrame(data=indices.numpy(), columns=['batch_idx', 'epoch_idx', 'sample_idx', 'row_idx'])
        features = pd.DataFrame(data=features.numpy().squeeze(), columns=dataset.columns)
        display(pd.concat([indices, features], axis=1).set_index(['batch_idx', 'epoch_idx', 'sample_idx', 'row_idx']))
```



#### Data Transformation & Parallelization of Data Producer and Data Consumer
```python
import itertools
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import tensorflow as tf

# Define Dataset
dataset = tf.random.normal(mean=100.0, stddev=9.0, shape=(100, 7)).numpy()
dataset = pd.DataFrame(dataset).add_prefix('COMP')

# Parameters for Parallelized Data Extraction
NUM_ROWS = dataset.shape[0]
BATCH_SIZE = 5
EPOCHS = 2

class CustomDataset(tf.data.Dataset):
    _BATCH_COUNTER = itertools.count()
    _EPOCHS_COUNTER = defaultdict(itertools.count)
    # OUTPUT: (indices, values)    
    OUTPUT_TYPES = (tf.dtypes.int32, tf.dtypes.float32)
    OUTPUT_SHAPES = ((4, ), (1, 7))
    
    def _generator(batch_idx, batch_size):
        epoch_idx = next(CustomDataset._EPOCHS_COUNTER[batch_idx])
        for sample_idx, (row_idx, row_series) in enumerate(dataset.iloc[batch_idx*batch_size:(batch_idx+1)*batch_size].iterrows()):
            yield ([batch_idx, epoch_idx, sample_idx, row_idx], [row_series.values])

    def __new__(cls, batch_size):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(next(cls._BATCH_COUNTER), batch_size),
            output_types=cls.OUTPUT_TYPES,
            output_shapes=cls.OUTPUT_SHAPES)

def Extraction(*args, **kwargs):
    tf.print("Data Extraction")
    return CustomDataset(BATCH_SIZE)

def tf_mapper(func):
    def wrapper(*args):
        return tf.py_function(func, inp=args, Tout=list(getattr(arg, 'dtype') for arg in args))
    return wrapper

@tf_mapper
def Preprocessing(indices, features):
    tf.print('Data Preprocessing')
    features = tf.linalg.normalize(features, axis=-1)[0]
    return indices, features

def IterableDataset(num_repeat=1):
    return tf.data.Dataset.range(num_repeat).interleave(Extraction, cycle_length=1).batch(BATCH_SIZE, drop_remainder=True).map(Preprocessing)

for batch_idx in range(NUM_ROWS//BATCH_SIZE):
    for indices, features in IterableDataset(num_repeat=EPOCHS):
        print('%-12s'%f'[BATCHIDX:{batch_idx}] ')    
        indices = pd.DataFrame(data=indices.numpy(), columns=['batch_idx', 'epoch_idx', 'sample_idx', 'row_idx'])
        features = pd.DataFrame(data=features.numpy().squeeze(), columns=dataset.columns)
        display(pd.concat([indices, features], axis=1).set_index(['batch_idx', 'epoch_idx', 'sample_idx', 'row_idx']))
```


#### Optimize pipeline performance
`DataFrame-Pipeline`
```python
import time
import itertools
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import tensorflow as tf

# Define Dataset
dataset = tf.random.normal(mean=100.0, stddev=9.0, shape=(100, 7)).numpy()
dataset = pd.DataFrame(dataset).add_prefix('COMP')

# Parameters for Parallelized Data Extraction
NUM_ROWS = dataset.shape[0]
BATCH_SIZE = 5
EPOCHS = 2

class CustomDataset(tf.data.Dataset):
    _BATCH_COUNTER = itertools.count()
    _EPOCHS_COUNTER = defaultdict(itertools.count)
    # OUTPUT: (indices, values)    
    OUTPUT_TYPES = (tf.dtypes.int32, tf.dtypes.float32)
    OUTPUT_SHAPES = ((4, ), (1, 7))
    
    def _generator(batch_idx, batch_size):
        epoch_idx = next(CustomDataset._EPOCHS_COUNTER[batch_idx])
        for sample_idx, (row_idx, row_series) in enumerate(dataset.iloc[batch_idx*batch_size:(batch_idx+1)*batch_size].iterrows()):
            yield ([batch_idx, epoch_idx, sample_idx, row_idx], [row_series.values])

    def __new__(cls, batch_size):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(next(cls._BATCH_COUNTER), batch_size),
            output_types=cls.OUTPUT_TYPES,
            output_shapes=cls.OUTPUT_SHAPES)

def Extraction(*args, **kwargs):
    #tf.print("Data Extraction")
    return CustomDataset(BATCH_SIZE)

def tf_mapper(func):
    def wrapper(*args):
        return tf.py_function(func, inp=args, Tout=list(getattr(arg, 'dtype') for arg in args))
    return wrapper

@tf_mapper
def Preprocessing(indices, features):
    #tf.print('Data Preprocessing')
    features = tf.linalg.normalize(features, axis=-1)[0]
    return indices, features

def IterableDataset_01(num_repeat=1):
    return tf.data.Dataset.range(num_repeat).interleave(Extraction, cycle_length=1).map(Preprocessing).batch(BATCH_SIZE, drop_remainder=True)
def IterableDataset_02(num_repeat=1):
    return tf.data.Dataset.range(num_repeat).interleave(Extraction, cycle_length=1).batch(BATCH_SIZE, drop_remainder=True).map(Preprocessing)
def IterableDataset_03(num_repeat=1):
    return tf.data.Dataset.range(num_repeat).interleave(Extraction, cycle_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True).map(Preprocessing)
def IterableDataset_04(num_repeat=1):
    return tf.data.Dataset.range(num_repeat).interleave(Extraction, cycle_length=1).batch(BATCH_SIZE, drop_remainder=True).map(Preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
def IterableDataset_05(num_repeat=1):
    return tf.data.Dataset.range(num_repeat).interleave(Extraction, cycle_length=1, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(BATCH_SIZE, drop_remainder=True).map(Preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
def IterableDataset_06(num_repeat=1):
    return tf.data.Dataset.range(num_repeat).interleave(Extraction, cycle_length=1).batch(BATCH_SIZE, drop_remainder=True).map(Preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache()
def IterableDataset_07(num_repeat=1):
    return tf.data.Dataset.range(num_repeat).interleave(Extraction, cycle_length=1).batch(BATCH_SIZE, drop_remainder=True).map(Preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().prefetch(tf.data.experimental.AUTOTUNE)
def IterableDataset_08(num_repeat=1):
    return tf.data.Dataset.range(num_repeat).interleave(Extraction, cycle_length=1).batch(BATCH_SIZE, drop_remainder=True).map(Preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().prefetch(tf.data.experimental.AUTOTUNE).unbatch()

def benchmark(iterable_dataset, name):
    start_time = time.perf_counter()
    for batch_idx in range(NUM_ROWS//BATCH_SIZE):
        for indices, features in iterable_dataset(num_repeat=EPOCHS):
            #print('%-12s'%f'[BATCHIDX:{batch_idx}] ')
            indices = pd.DataFrame(data=indices.numpy(), columns=['batch_idx', 'epoch_idx', 'sample_idx', 'row_idx'])
            features = pd.DataFrame(data=features.numpy().squeeze(), columns=dataset.columns)
            #display(pd.concat([indices, features], axis=1).set_index(['batch_idx', 'epoch_idx', 'sample_idx', 'row_idx']))
    exec_time = time.perf_counter() - start_time            
    tf.print(f"%-{90}s"%f"[처리 과정에 따른 실행 시간][{name}]", ':', exec_time)
    return name, exec_time

report = list()
report.append(benchmark(IterableDataset_01, name='Scalar Sequential Mapping'))
report.append(benchmark(IterableDataset_02, name='Vectorizing Sequential Mapping'))
report.append(benchmark(IterableDataset_03, name='Vectorizing Sequential Mapping wtih Parallel Interleave'))
report.append(benchmark(IterableDataset_04, name='Vectorizing Parallel Mapping'))
report.append(benchmark(IterableDataset_05, name='Vectorizing Parallel Mapping wtih Parallel Interleave'))
report.append(benchmark(IterableDataset_06, name='Caching Vectorizing Parallel Mapping'))
report.append(benchmark(IterableDataset_07, name='Caching Vectorizing Parallel Mapping & Prefetching'))
report.append(benchmark(IterableDataset_08, name='Caching Vectorizing Parallel Mapping & Prefetching & Unbatching'))
report = pd.DataFrame(report, columns=['Name', 'ExecTime']).set_index('Name')
report['Rank'] = report.rank()
report.sort_values(by='Rank')
```

#### Evalutate Timelines
```python
import itertools
from collections import defaultdict
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

def draw_timeline(timeline, title, width=0.5, annotate=False, save=False):
    # 타임라인에서 유효하지 않은 항목(음수 또는 빈 스텝) 제거
    invalid_mask = np.logical_and(timeline['times'] > 0, timeline['steps'] != b'')[:,0]
    steps = timeline['steps'][invalid_mask].numpy()
    times = timeline['times'][invalid_mask].numpy()
    values = timeline['values'][invalid_mask].numpy()

    # 처음 발견될 때 순서대로 다른 스텝을 가져옵니다.
    step_ids, indices = np.stack(np.unique(steps, return_index=True))
    step_ids = step_ids[np.argsort(indices)]

    # 시작 시간을 0으로 하고 최대 시간 값을 계산하십시오.
    min_time = times[:,0].min()
    times[:,0] = (times[:,0] - min_time)
    end = max(width, (times[:,0]+times[:,1]).max() + 0.01)

    cmap = mpl.cm.get_cmap("plasma")
    plt.close()
    fig, axs = plt.subplots(len(step_ids), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle(title)
    fig.set_size_inches(25.0, len(step_ids))
    plt.xlim(-0.01, end)

    for i, step in enumerate(step_ids):
        step_name = step.decode()
        ax = axs[i]
        ax.set_ylabel(step_name)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("time (s)")
        ax.set_xticklabels([])
        ax.grid(which="both", axis="x", color="k", linestyle=":")

        # 주어진 단계에 대한 타이밍과 주석 얻기
        entries_mask = np.squeeze(steps==step)
        serie = np.unique(times[entries_mask], axis=0)
        annotations = values[entries_mask]

        ax.broken_barh(serie, (0, 1), color=cmap(i / len(step_ids)), linewidth=1, alpha=0.66)
        if annotate:
            for j, (start, width) in enumerate(serie):
                annotation = "\n".join([f"{l}: {v}" for l,v in zip(("i", "e", "s"), annotations[j])])
                ax.text(start + 0.001 + (0.001 * (j % 2)), 0.55 - (0.1 * (j % 2)), annotation,
                        horizontalalignment='left', verticalalignment='center')
    if save:
        plt.savefig(title.lower().translate(str.maketrans(" ", "_")) + ".svg")

class TimeMeasuredDataset(tf.data.Dataset):
    # 출력: (steps, timings, counters)
    OUTPUT_TYPES = (tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int32)
    OUTPUT_SHAPES = ((2, 1), (2, 2), (2, 3))

    _INSTANCES_COUNTER = itertools.count()  # 생성된 데이터셋 수
    _EPOCHS_COUNTER = defaultdict(itertools.count)  # 각 데이터를 수행한 에포크 수

    def _generator(instance_idx, num_samples):
        epoch_idx = next(TimeMeasuredDataset._EPOCHS_COUNTER[instance_idx])

        # 파일 열기
        open_enter = time.perf_counter()
        time.sleep(0.03)
        open_elapsed = time.perf_counter() - open_enter

        for sample_idx in range(num_samples):
            # 파일에서 데이터(줄, 기록) 읽어오기
            read_enter = time.perf_counter()
            time.sleep(0.015)
            read_elapsed = time.perf_counter() - read_enter

            yield (
                [("Open",), ("Read",)],
                [(open_enter, open_elapsed), (read_enter, read_elapsed)],
                [(instance_idx, epoch_idx, -1), (instance_idx, epoch_idx, sample_idx)]
            )
            open_enter, open_elapsed = -1., -1.  # 음수는 필터링됨

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=cls.OUTPUT_TYPES,
            output_shapes=cls.OUTPUT_SHAPES,
            args=(next(cls._INSTANCES_COUNTER), num_samples)
        )

_batch_map_num_items = 10
def dataset_generator_fun(*args):
    print('extraction')
    return TimeMeasuredDataset(num_samples=_batch_map_num_items)    
    
def map_decorator(func):
    def wrapper(steps, times, values):
        # 자동 그래프가 메서드를 컴파일하지 못하도록 tf.py_function을 사용
        return tf.py_function(
            func,
            inp=(steps, times, values),
            Tout=(steps.dtype, times.dtype, values.dtype)
        )
    return wrapper

@map_decorator
def naive_map(steps, times, values):
    map_enter = time.perf_counter()
    time.sleep(0.001)  # 시간 소비 스텝
    time.sleep(0.0001)  # 메모리 소비 스텝
    map_elapsed = time.perf_counter() - map_enter
    return (
        tf.concat((steps, [["Map"]]), axis=0),
        tf.concat((times, [[map_enter, map_elapsed]]), axis=0),
        tf.concat((values, [values[-1]]), axis=0)
    )

@map_decorator
def time_consumming_map(steps, times, values):
    map_enter = time.perf_counter()
    time.sleep(0.001 * values.shape[0])  # 시간 소비 스텝
    map_elapsed = time.perf_counter() - map_enter

    return (
        tf.concat((steps, tf.tile([[["1st map"]]], [steps.shape[0], 1, 1])), axis=1),
        tf.concat((times, tf.tile([[[map_enter, map_elapsed]]], [times.shape[0], 1, 1])), axis=1),
        tf.concat((values, tf.tile([[values[:][-1][0]]], [values.shape[0], 1, 1])), axis=1)
    )

@map_decorator
def memory_consumming_map(steps, times, values):
    map_enter = time.perf_counter()
    time.sleep(0.0001 * values.shape[0])  # 메모리 소비 스텝
    map_elapsed = time.perf_counter() - map_enter
    # 배치 차원을 다루는 데 tf.tile 사용
    return (
        tf.concat((steps, tf.tile([[["2nd map"]]], [steps.shape[0], 1, 1])), axis=1),
        tf.concat((times, tf.tile([[[map_enter, map_elapsed]]], [times.shape[0], 1, 1])), axis=1),
        tf.concat((values, tf.tile([[values[:][-1][0]]], [values.shape[0], 1, 1])), axis=1)
    )


def timelined_benchmark(dataset, num_epochs=2):
    # 누산기 초기화
    steps_acc = tf.zeros([0, 1], dtype=tf.dtypes.string)
    times_acc = tf.zeros([0, 2], dtype=tf.dtypes.float32)
    values_acc = tf.zeros([0, 3], dtype=tf.dtypes.int32)

    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        epoch_enter = time.perf_counter()
        for (steps, times, values) in dataset:
            # 데이터셋 준비 정보 기록하기
            steps_acc = tf.concat((steps_acc, steps), axis=0)
            times_acc = tf.concat((times_acc, times), axis=0)
            values_acc = tf.concat((values_acc, values), axis=0)

            # 훈련 시간 시뮬레이션
            train_enter = time.perf_counter()
            time.sleep(0.01)
            train_elapsed = time.perf_counter() - train_enter

            # 훈련 정보 기록하기
            steps_acc = tf.concat((steps_acc, [["Train"]]), axis=0)
            times_acc = tf.concat((times_acc, [(train_enter, train_elapsed)]), axis=0)
            values_acc = tf.concat((values_acc, [values[-1]]), axis=0)

        epoch_elapsed = time.perf_counter() - epoch_enter
        # 에포크 정보 기록하기
        steps_acc = tf.concat((steps_acc, [["Epoch"]]), axis=0)
        times_acc = tf.concat((times_acc, [(epoch_enter, epoch_elapsed)]), axis=0)
        values_acc = tf.concat((values_acc, [[-1, epoch_num, -1]]), axis=0)
        time.sleep(0.001)

    tf.print("실행 시간:", time.perf_counter() - start_time)
    return {"steps": steps_acc, "times": times_acc, "values": values_acc}        


naive_timeline = timelined_benchmark(
    tf.data.Dataset.range(2)
    .flat_map(dataset_generator_fun)
    .map(naive_map)
    .batch(_batch_map_num_items, drop_remainder=True)
    .unbatch(),
    5
)

optimized_timeline = timelined_benchmark(
    tf.data.Dataset.range(2)
    .interleave(  # 데이터 읽기 병렬화
        dataset_generator_fun,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(  # 매핑된 함수 벡터화
        _batch_map_num_items,
        drop_remainder=True)
    .map(  # 맵 변환 병렬화
        time_consumming_map,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .cache()  # 데이터 캐시
    .map(  # 메모리 사용량 줄이기
        memory_consumming_map,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .prefetch(  # 프로듀서와 컨슈머 작업 오버랩
        tf.data.experimental.AUTOTUNE
    )
    .unbatch(),
    5
)

draw_timeline(naive_timeline, title="Naive", width=10, save=False)
draw_timeline(optimized_timeline, title="Optimized", width=10, save=False)
```



### Real Use Cases

#### Loading Dataframe
```python
import tensorflow as tf
from ailever.dataset import SKAPI

df = SKAPI.housing()
df = df.copy()

target = df.pop('target')
iterable_dataset = tf.data.Dataset.from_tensor_slices((dict(df), target)).shuffle(buffer_size=df.shape[0]).batch(5).take(1)
dataset_iterator = iter(iterable_dataset)
features, targets = dataset_iterator.get_next()

print(features.keys())
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

```python
import tensorflow as tf
import tensorflow_datasets as tfds
from ailever.dataset import SKAPI

df = SKAPI.housing()
df = df.copy()

target = df.pop('target')
dataset = tf.data.Dataset.from_tensor_slices((df, target)).shuffle(buffer_size=df.shape[0])

# tensor-return
for features, targets in dataset.batch(5).take(1):
    print(features)
    print(targets)

# ndarray-return    
for features, targets in tfds.as_numpy(dataset.batch(5).take(1)):
    print(features)
    print(targets)    
```

#### Preprocessing DataFrame
```python
from ailever.dataset import SKAPI
import tensorflow as tf

# dataset
dataset = SKAPI.housing()
target = dataset.pop('target')

def preprocessing(x, y):
    x = tf.cast(x, tf.float64)
    y = tf.cast(y, tf.float32)
    return x, y

iterable_dataset = tf.data.Dataset.from_tensor_slices((dataset, target)).shuffle(buffer_size=target.shape[0]).batch(4).map(preprocessing).take(1)
dataset_iterator = iter(iterable_dataset)
x, y = dataset_iterator.get_next()
x, y
```

#### Realistic Usage
`Sequential API`
```python
from ailever.dataset import SKAPI
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# dataset
dataset = SKAPI.housing()
target = dataset.pop('target')
train_X, test_X, train_y, test_y = train_test_split(dataset, target, test_size=0.3, shuffle=False)

iterable_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).shuffle(buffer_size=train_X.shape[0]).batch(4)
iterable_feature = tf.data.Dataset.from_tensor_slices(test_X).batch(4)

# train
model = models.Sequential()
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dense(10, activation="relu"))
model.add(layers.Dense(1))
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.fit(iterable_dataset, epochs=2)    

# inference
model.predict(iterable_feature)
model.predict(iterable_feature, steps = 50)
#model.predict(iterable_dataset)
#model.predict(iterable_dataset, steps = 50)

# evaluation
loss, metrics = model.evaluate(iterable_dataset)
loss, metrics = model.evaluate(iterable_dataset, steps=50)
```

`Model Class`
```python
import tensorflow as tf
from tensorflow.keras import models
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
class Architecture(models.Model):
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

model = Architecture()
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
print(model.summary())
tf.keras.utils.plot_model(model, show_shapes=True)
```




<br><br><br>

---


## Models
### Linear Regression
#### Simple Linear Regression
`with gradient implementation`
```python
import tensorflow as tf

# Data
X = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
Y = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)

# W, b initialize
W = tf.Variable(2.)
b = tf.Variable(1.)

# W, b update
learning_rate = 0.01
for i in range(2000):
    # forward
    hypothesis = W * X + b
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    # backward
    W_grad = tf.reduce_mean(2*tf.multiply(tf.multiply(W, X) + b - Y, X))
    b_grad = tf.reduce_mean(2*tf.multiply(tf.multiply(W, X) + b - Y, 1))    
    W.assign(W - tf.multiply(learning_rate, W_grad))
    b.assign(b - tf.multiply(learning_rate, b_grad))
    
    if i % 100 == 0:
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

# predict
print(W.numpy(), b.numpy())
```
`with gradient tape`
```python
import tensorflow as tf

# Data
X = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
Y = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)

# W, b initialize
W = tf.Variable(2.9)
b = tf.Variable(0.5)

# W, b update
learning_rate = 0.01
for i in range(100):
    # forward
    with tf.GradientTape() as tape:
        hypothesis = W * X + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    # backward
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

# predict
print(W.numpy(), b.numpy())
```

#### Multi-variate Linear Regression

`Vector Operation`
```python
import tensorflow as tf 

X1 = [1, 0, 3, 0, 5]
X2 = [0, 2, 0, 4, 0]
Y  = [1, 2, 3, 4, 5]

W1 = tf.Variable(tf.random.uniform((1,), -10.0, 10.0))
W2 = tf.Variable(tf.random.uniform((1,), -10.0, 10.0))
b  = tf.Variable(tf.random.uniform((1,), -10.0, 10.0))

learning_rate = tf.Variable(0.001)
for i in range(1000+1):
    # forward
    with tf.GradientTape() as tape:
        hypothesis = W1 * X1 + W2 * X2 + b
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # backward
    W1_grad, W2_grad, b_grad = tape.gradient(cost, [W1, W2, b])
    W1.assign_sub(learning_rate * W1_grad)
    W2.assign_sub(learning_rate * W2_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 50 == 0:
        print("{:5} | {:10.6f} | {:10.4f} | {:10.4f} | {:10.6f}".format(
          i, cost.numpy(), W1.numpy()[0], W2.numpy()[0], b.numpy()[0]))

print(W1.numpy()[0], W2.numpy()[0], b.numpy()[0])
```

`Matrix Column-based Operation`
```python
import tensorflow as tf 

X = [[1., 0., 3., 0., 5.],   # X1
     [0., 2., 0., 4., 0.]]   # X2
Y  = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random.uniform((1, 2), -1.0, 1.0))
b = tf.Variable(tf.random.uniform((1,), -1.0, 1.0))

learning_rate = tf.Variable(0.001)

for i in range(1000+1):
    # forward
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(W, X) + b # (1, 2) * (2, 5) = (1, 5)
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # backward
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    
    if i % 50 == 0:
        print("{:5} | {:10.6f} | {:10.4f} | {:10.4f} | {:10.6f}".format(
            i, cost.numpy(), W.numpy()[0][0], W.numpy()[0][1], b.numpy()[0]))

print(W.numpy()[0][0], W.numpy()[0][1], b.numpy()[0])            
```
```python
import tensorflow as tf

# 앞의 코드에서 bias(b)를 행렬에 추가
X = [[1., 1., 1., 1., 1.], # bias(b)
     [1., 0., 3., 0., 5.], # X1 
     [0., 2., 0., 4., 0.]] # X2
Y = [1, 2, 3, 4, 5]

W = tf.Variable(tf.random.uniform((1, 3), -1.0, 1.0)) # [1, 3]으로 변경하고, b 삭제

learning_rate = 0.001
optimizer = tf.keras.optimizers.SGD(learning_rate)
for i in range(1000+1):
    # forward
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(W, X)
        cost = tf.reduce_mean(tf.square(hypothesis - Y))

    # backward
    grads = tape.gradient(cost, [W])
    optimizer.apply_gradients(grads_and_vars=zip(grads, [W]))
    if i % 50 == 0:
        print("{:5} | {:10.6f} | {:10.4f} | {:10.4f} | {:10.4f}".format(
            i, cost.numpy(), W.numpy()[0][0], W.numpy()[0][1], W.numpy()[0][2]))

print(W.numpy()[0][0], W.numpy()[0][1], W.numpy()[0][2]) # b, W1, W2 / Y = W1*X1 + W2*X2 + b
```

`Matrix Row-based Operation`
```python
import tensorflow as tf

data = tf.constant(   
     # X1,   X2,    X3,   y
    [[ 73.,  80.,  75., 152. ],
     [ 93.,  88.,  93., 185. ],
     [ 89.,  91.,  90., 180. ],
     [ 96.,  98., 100., 196. ],
     [ 73.,  66.,  70., 142. ]], dtype=tf.float32)
X = data[:, :-1]
y = data[:, -1][:, tf.newaxis]

W = tf.Variable(tf.random.normal((3, 1)))
b = tf.Variable(tf.random.normal((1,)))

learning_rate = 0.000001
for i in range(2000+1):
    # forward
    with tf.GradientTape() as tape:
        hypothesis = tf.matmul(X, W) + b # (5,3) * (3,1)
        cost = tf.reduce_mean((tf.square(hypothesis - y)))
    
    # backward
    W_grad, b_grad = tape.gradient(cost, [W, b])
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    
    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))
print(W.numpy().squeeze(), b.numpy())
```

### Binary-Class Classification
#### Binary Classification(Logistic Regression)
`logistic`
```python
import tensorflow as tf 

def forward(x_train, params:list):
    W = params[0]
    b = params[1]
    #hypothesis = tf.divide(1., 1. + tf.exp(-(tf.matmul(x_train, W) + b)))
    hypothesis = tf.sigmoid(tf.matmul(x_train, W) + b)
    return hypothesis

def loss_fn(hypothesis, target):
    cost = -tf.reduce_mean(target * tf.math.log(hypothesis) + (1 - target) * tf.math.log(1 - hypothesis))
    return cost

def metric(hypothesis, target):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, target), dtype=tf.int32))
    return accuracy

W = tf.Variable(tf.zeros([2,1]), trainable=True, name='weight')
b = tf.Variable(tf.zeros([1]), trainable=True, name='bias')
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

     #X1, X2 
X = tf.constant(
    [[1., 2.],
     [2., 3.],
     [3., 1.],
     [4., 3.],
     [5., 3.],
     [6., 2.]])
Y = tf.constant(
    [[0.],
     [0.],
     [0.],
     [1.],
     [1.],
     [1.]])
dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(len(X))#.repeat()

for step in range(1001):
    for x_train, target in iter(dataset):
        # forward
        with tf.GradientTape() as tape:
            hypothesis = forward(x_train, params=[W, b])
            cost = loss_fn(hypothesis, target)

        # backward
        grads = tape.gradient(cost, [W, b])
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W,b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss_fn(hypothesis, target)))
test_acc = metric(hypothesis, target)
print("Testset Accuracy: {:.4f}".format(test_acc))
```

`softmax`
```python
import tensorflow as tf 
from tensorflow.keras import Model

class Architecture(Model):
    def __init__(self):
        super(Architecture, self).__init__()
        self.W = tf.Variable(tf.random.normal((4, 2)), name='weight')
        self.b = tf.Variable(tf.random.normal((2,)), name='bias')
        
    def forward(self, X):
        return tf.nn.softmax(tf.matmul(X, self.W) + self.b)
    
    def loss_fn(self, hypothesis, target):
        cost = tf.reduce_mean(-tf.reduce_sum(target * tf.math.log(hypothesis), axis=1))        
        return cost
    
    def grad_fn(self, x_train, target):
        with tf.GradientTape() as tape:
            hypothesis = self.forward(x_train)
            cost = self.loss_fn(hypothesis, target)
        grads = tape.gradient(cost, self.variables)            
        return grads
    
    def fit(self, x_train, target, epochs=5000, verbose=500):
        optimizer =  tf.keras.optimizers.SGD(learning_rate=0.1)
        for i in range(epochs):
            grads = self.grad_fn(x_train, target)
            optimizer.apply_gradients(zip(grads, self.variables))
            if (i==0) | ((i+1)%verbose==0):
                hypothesis = self.forward(x_train)
                print('Loss at epoch %d: %f' %(i+1, self.loss_fn(hypothesis, target).numpy()))

X = tf.constant(
    [[1., 2., 1., 1.],
     [2., 1., 3., 2.],
     [3., 1., 3., 4.],
     [4., 1., 5., 5.],
     [1., 7., 5., 5.],
     [1., 2., 5., 6.],
     [1., 6., 6., 6.],
     [1., 7., 7., 7.]])
Y = tf.constant(
    [[1., 0.],
     [1., 0.],
     [1., 0.],
     [1., 0.],
     [0., 1.],
     [0., 1.],
     [0., 1.],
     [0., 1.]])

model = Architecture()
model.fit(X, Y)
```

### Multi-Class Classification
`softmax`
```python
import tensorflow as tf 
from tensorflow.keras import Model

class Architecture(Model):
    def __init__(self):
        super(Architecture, self).__init__()
        self.W = tf.Variable(tf.random.normal((4, 3)), name='weight')
        self.b = tf.Variable(tf.random.normal((3,)), name='bias')
        
    def forward(self, X):
        return tf.nn.softmax(tf.matmul(X, self.W) + self.b)
    
    def loss_fn(self, hypothesis, target):
        cost = tf.reduce_mean(-tf.reduce_sum(target * tf.math.log(hypothesis), axis=1))        
        return cost
    
    def grad_fn(self, x_train, target):
        with tf.GradientTape() as tape:
            hypothesis = self.forward(x_train)
            cost = self.loss_fn(hypothesis, target)
        grads = tape.gradient(cost, self.variables)            
        return grads
    
    def fit(self, x_train, target, epochs=5000, verbose=500):
        optimizer =  tf.keras.optimizers.SGD(learning_rate=0.1)
        for i in range(epochs):
            grads = self.grad_fn(x_train, target)
            optimizer.apply_gradients(zip(grads, self.variables))
            if (i==0) | ((i+1)%verbose==0):
                hypothesis = self.forward(x_train)
                print('Loss at epoch %d: %f' %(i+1, self.loss_fn(hypothesis, target).numpy()))

X = tf.constant(
    [[1., 2., 1., 1.],
     [2., 1., 3., 2.],
     [3., 1., 3., 4.],
     [4., 1., 5., 5.],
     [1., 7., 5., 5.],
     [1., 2., 5., 6.],
     [1., 6., 6., 6.],
     [1., 7., 7., 7.]])
Y = tf.constant(
    [[0., 0., 1.],
     [0., 0., 1.],
     [0., 0., 1.],
     [0., 1., 0.],
     [0., 1., 0.],
     [0., 1., 0.],
     [1., 0., 0.],
     [1., 0., 0.]])

model = Architecture()
model.fit(X, Y)
```


### Multi-Label Classification
```python
import tensorflow as tf 
from tensorflow.keras import Model

class Architecture(Model):
    def __init__(self):
        super(Architecture, self).__init__()
        self.W = tf.Variable(tf.random.normal((4, 3)), name='weight')
        self.b = tf.Variable(tf.random.normal((3,)), name='bias')
        
    def forward(self, X):
        return tf.divide(1, 1+tf.math.exp(-(tf.matmul(X, self.W) + self.b)))
    
    def loss_fn(self, hypothesis, target):
        cost = -tf.reduce_mean(target * tf.math.log(hypothesis) + (1 - target) * tf.math.log(1 - hypothesis))
        return cost
    
    def grad_fn(self, x_train, target):
        with tf.GradientTape() as tape:
            hypothesis = self.forward(x_train)
            cost = self.loss_fn(hypothesis, target)
        grads = tape.gradient(cost, self.variables)            
        return grads
    
    def fit(self, x_train, target, epochs=10000, verbose=500):
        optimizer =  tf.keras.optimizers.Adam(learning_rate=0.1)
        for i in range(epochs):
            grads = self.grad_fn(x_train, target)
            optimizer.apply_gradients(zip(grads, self.variables))
            if (i==0) | ((i+1)%verbose==0):
                hypothesis = self.forward(x_train)
                print('Loss at epoch %d: %f' %(i+1, self.loss_fn(hypothesis, target).numpy()))

X = tf.constant(
    [[1., 2., 1., 1.],
     [2., 1., 3., 2.],
     [3., 1., 3., 4.],
     [4., 1., 5., 5.],
     [1., 7., 5., 5.],
     [1., 2., 5., 6.],
     [1., 6., 6., 6.],
     [1., 7., 7., 7.]])
Y = tf.constant(
    [[0., 0., 0.],
     [1., 0., 0.],
     [0., 1., 0.],
     [0., 0., 1.],
     [1., 1., 0.],
     [1., 0., 1.],
     [0., 1., 1.],
     [1., 1., 1.]])

model = Architecture()
model.fit(X, Y)
```

<br><br><br>

---


## Neural Network

<br><br><br>

---


## Convolutnal Neural Network
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models 
from tensorflow.keras import datasets
from tensorflow.keras import utils
import numpy as np
import matplotlib.pyplot as plt
import os

tf.random.set_seed(777)

class Architecture(models.Model):
    def __init__(self):
        super(Architecture, self).__init__()
        self.conv1 = layers.Conv2D(filters=32, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        self.pool1 = layers.MaxPool2D(padding='SAME')
        self.conv2 = layers.Conv2D(filters=64, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        self.pool2 = layers.MaxPool2D(padding='SAME')
        self.conv3 = layers.Conv2D(filters=128, kernel_size=[3, 3], padding='SAME', activation=tf.nn.relu)
        self.pool3 = layers.MaxPool2D(padding='SAME')
        self.pool3_flat = layers.Flatten()
        self.dense4 = layers.Dense(units=256, activation=tf.nn.relu)
        self.drop4 = layers.Dropout(rate=0.4)
        self.dense5 = layers.Dense(units=10)
    def call(self, inputs, training=False):
        net = self.conv1(inputs)
        net = self.pool1(net)
        net = self.conv2(net)
        net = self.pool2(net)
        net = self.conv3(net)
        net = self.pool3(net)
        net = self.pool3_flat(net)
        net = self.dense4(net)
        net = self.drop4(net)
        net = self.dense5(net)
        return net

def loss_fn(model, images, labels):
    logits = model(images, training=True)
    loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
        y_pred=logits, y_true=labels, from_logits=True))
    return loss   

def grad_fn(model, images, labels):
    with tf.GradientTape() as tape:
        loss = loss_fn(model, images, labels)
    return tape.gradient(loss, model.variables)

def evaluate(models, images, labels):
    predictions = np.zeros_like(labels)
    for model in models:
        logits = model(images, training=False)
        predictions += logits
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()    
train_images = train_images.astype(np.float32) / 255.
test_images = test_images.astype(np.float32) / 255.
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)
train_labels = utils.to_categorical(train_labels, 10)
test_labels = utils.to_categorical(test_labels, 10)
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(buffer_size=100000).batch(100)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(100)

models = []
num_models = 3
for m in range(num_models):
    models.append(Architecture())

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# train my model
print('Learning started. It takes sometime.')
training_epochs = 15
for epoch in range(training_epochs):
    avg_loss = 0.
    avg_train_acc = 0.
    avg_test_acc = 0.
    train_step = 0
    test_step = 0    
    for images, labels in train_dataset:
        for model in models:
            #train(model, images, labels)
            grads = grad_fn(model, images, labels)                
            optimizer.apply_gradients(zip(grads, model.variables))
            loss = loss_fn(model, images, labels)
            avg_loss += loss / num_models
        acc = evaluate(models, images, labels)
        avg_train_acc += acc
        train_step += 1
    avg_loss = avg_loss / train_step
    avg_train_acc = avg_train_acc / train_step
    
    for images, labels in test_dataset:        
        acc = evaluate(models, images, labels)        
        avg_test_acc += acc
        test_step += 1    
    avg_test_acc = avg_test_acc / test_step    

    print('Epoch:', '{}'.format(epoch + 1), 'loss =', '{:.8f}'.format(avg_loss), 
          'train accuracy = ', '{:.4f}'.format(avg_train_acc), 
          'test accuracy = ', '{:.4f}'.format(avg_test_acc))
    
print('Learning Finished!')
```
<br><br><br>


---


## Recurrent Neural Network

<br><br><br>

---

## Keras API

### Layer
- https://www.tensorflow.org/api_docs/python/tf/keras/layers
- https://www.tensorflow.org/api_docs/python/tf/keras/activations
- https://www.tensorflow.org/api_docs/python/tf/keras/initializers
- https://www.tensorflow.org/api_docs/python/tf/keras/Model
- https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
- https://www.tensorflow.org/api_docs/python/tf/keras/Input

```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import utils

model = models.Sequential()
model.add(layers.Input(shape=(16,)))
model.add(layers.Dense(units=32, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

input_tensor = tf.random.normal(shape=(10,16))
layer = model.layers[-1]
output_tensor = layer(input_tensor)

print('[IN/OUT]:', input_tensor.shape, output_tensor.shape)
print('[LAYER]:', layer.weights[0].shape, layer.weights[0].shape)
print()

model.summary()
display(utils.plot_model(model, to_file="model.png", show_layer_names=True, show_shapes=True))
```

`browser mode`
```python
import os
import subprocess
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Input(shape=(16,)))
model.add(layers.Dense(units=32, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

input_tensor = tf.random.normal(shape=(10,16))
layer = model.layers[-1]
output_tensor = layer(input_tensor)

# tensorboard
tensorboard_rootpath = "dense"
tensorboard_logpath = os.path.join(tensorboard_rootpath, datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(tensorboard_logpath, histogram_freq=1)
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.fit(x=input_tensor, y=output_tensor, validation_data=None, epochs=5, callbacks=[tensorboard_callback])
server = subprocess.Popen(["tensorboard", "--logdir", f"{tensorboard_rootpath}", "--port=6006"])
#server.kill()

print('[IN/OUT]:', input_tensor.shape, output_tensor.shape)
print('[LAYER]:', layer.weights[0].shape, layer.weights[0].shape)
print()
```

`jupyter mode`
```python
%load_ext tensorboard
tensorboard_rootpath = "layer" # CELL COMMAND at the bottom: %tensorboard --logdir layer 

import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Input(shape=(16,)))
model.add(layers.Dense(units=32, activation='relu', use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))

input_tensor = tf.random.normal(shape=(10,16))
layer = model.layers[-1]
output_tensor = layer(input_tensor)

# tensorboard
tensorboard_logpath = os.path.join(tensorboard_rootpath, datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(tensorboard_logpath, histogram_freq=1)
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.fit(x=input_tensor, y=output_tensor, validation_data=None, epochs=5, callbacks=[tensorboard_callback])

print('[IN/OUT]:', input_tensor.shape, output_tensor.shape)
print('[LAYER]:', layer.weights[0].shape, layer.weights[0].shape)
print()

%tensorboard --logdir layer
```

<br><br><br>


### Optimizer
- https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
  
`Custom`
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
lr_schedule = CustomSchedule(initial_learning_rate=0.1, gradients=gradients)

# Assgin gradient policy to trainable tensor
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
optimizer.apply_gradients(zip(gradients, [W])) # optimizer.weights[-1], optimizer.get_weights(), optimizer.set_weights(optimizer.weights)
W.numpy()
```
`seperated process into (1)computing gradient and (2)updating it [with GradientTape]`
```python
import tensorflow as tf
from tensorflow.keras import optimizers

W = tf.Variable(4.0, trainable=True)
Y = tf.Variable(1.0, trainable=False)
with tf.GradientTape() as tape:
    cost = (W - Y)**2

gradients = tape.gradient(cost, [W]); print('[gradient]:', gradients[0].numpy())

lr_schedule = optimizers.schedules.PolynomialDecay(initial_learning_rate=0.1, decay_steps=10000, end_learning_rate=0.0001, power=1.0, cycle=False, name=None)
lr_schedule = optimizers.schedules.PiecewiseConstantDecay(boundaries=[100000, 110000], values=[0.1, 0.5, 0.1], name=None)
lr_schedule = optimizers.schedules.InverseTimeDecay(initial_learning_rate=1e-1, decay_steps=10000, decay_rate=0.9, staircase=False, name=None)
lr_schedule = optimizers.schedules.CosineDecay(initial_learning_rate=1e-1, decay_steps=10000, alpha=0.0, name=None)
lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1, decay_steps=10000, decay_rate=0.9, name=None)
#config = optimizers.schedules.serialize(lr_schedule)
#lr_schedule = optimizers.schedules.deserialize(config)

optimizer = optimizers.SGD(learning_rate=lr_schedule, momentum=0.0, nesterov=False, name='SGD')
optimizer = optimizers.RMSprop(learning_rate=lr_schedule, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop')
optimizer = optimizers.Adagrad(learning_rate=lr_schedule, initial_accumulator_value=0.1, epsilon=1e-07, name='Adagrad')
optimizer = optimizers.Adadelta(learning_rate=lr_schedule, rho=0.95, epsilon=1e-07, name='Adadelta')
optimizer = optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
optimizer = optimizers.Adamax(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax')
optimizer = optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
optimizer = optimizers.Ftrl(learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0)
#config = optimizers.serialize(optimizer)
#optimizer = optimizers.deserialize(config)

optimizer.apply_gradients(zip(gradients, [W])) # optimizer.weights[-1], optimizer.get_weights(), optimizer.set_weights(optimizer.weights)
W.numpy()
```
`all in one process [within computing gradient and updating it]`
```python
import tensorflow as tf
from tensorflow.keras import optimizers

W = tf.Variable(4.0, trainable=True)
Y = tf.Variable(1.0, trainable=False)
trainable_list_fn = lambda : W
loss_fn = lambda: (W - Y)**2

lr_schedule = optimizers.schedules.PolynomialDecay(initial_learning_rate=0.1, decay_steps=10000, end_learning_rate=0.0001, power=1.0, cycle=False, name=None)
lr_schedule = optimizers.schedules.PiecewiseConstantDecay(boundaries=[100000, 110000], values=[0.1, 0.5, 0.1], name=None)
lr_schedule = optimizers.schedules.InverseTimeDecay(initial_learning_rate=1e-1, decay_steps=10000, decay_rate=0.9, staircase=False, name=None)
lr_schedule = optimizers.schedules.CosineDecay(initial_learning_rate=1e-1, decay_steps=10000, alpha=0.0, name=None)
lr_schedule = optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-1, decay_steps=10000, decay_rate=0.9, name=None)
#config = optimizers.schedules.serialize(lr_schedule)
#lr_schedule = optimizers.schedules.deserialize(config)

optimizer = optimizers.SGD(learning_rate=lr_schedule, momentum=0.0, nesterov=False, name='SGD')
optimizer = optimizers.RMSprop(learning_rate=lr_schedule, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop')
optimizer = optimizers.Adagrad(learning_rate=lr_schedule, initial_accumulator_value=0.1, epsilon=1e-07, name='Adagrad')
optimizer = optimizers.Adadelta(learning_rate=lr_schedule, rho=0.95, epsilon=1e-07, name='Adadelta')
optimizer = optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
optimizer = optimizers.Adamax(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax')
optimizer = optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
optimizer = optimizers.Ftrl(learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0)
#config = optimizers.serialize(optimizer)
#optimizer = optimizers.deserialize(config)

optimizer.minimize(loss_fn, trainable_list_fn)

W.numpy()
```



<br><br><br>

### Cost Function
- https://www.tensorflow.org/api_docs/python/tf/keras/losses
  
`Custom`
```python
import tensorflow as tf
from tensorflow.keras import losses

class CustomLoss(losses.Loss):
    def call(self, target, hypothesis):
        target = tf.cast(target, hypothesis.dtype)
        return tf.math.reduce_mean(tf.math.square(hypothesis - target), axis=-1)

target = tf.constant([[0., 1.], [0., 0.]])
hypothesis = tf.constant([[1., 1.], [1., 0.]])
cost = CustomLoss()
cost(target, hypothesis) 
```
```python
from tensorflow.keras import losses

target = [0, 1, 1, 1]
hypothesis = [0.3, 0.8, 0.5, 0.2]
cost = losses.BinaryCrossentropy(from_logits=False)
cost(target, hypothesis) 
tf.math.reduce_mean(-tf.math.log([(1-0.3), 0.8, 0.5, 0.2]))                             # from_logits=False
tf.math.reduce_mean(-tf.math.log([(1-0.57444252), 0.68997448, 0.62245933, 0.549834  ])) # from_logits=True, 1/(1 + np.exp(-1*np.array([0.3, 0.8, 0.5, 0.2])))

target = [[0, 1, 0], [0, 0, 1]]
hypothesis = [[0.05, 0.95, 0], [0.2, 0.7, 0.1]]
cost = losses.CategoricalCrossentropy(from_logits=False)
cost(target, hypothesis)
tf.math.reduce_mean(-tf.math.log([0.95, 0.1]))                                                            # from_logits=False
tf.math.reduce_mean(-tf.math.log([tf.nn.softmax([0.05, 0.95, 0])[1], tf.nn.softmax([0.2, 0.7, 0.1])[2]])) # from_logits=True

target = [1, 2]
hypothesis = [[0.05, 0.95, 0], [0.2, 0.7, 0.1]]
cost = losses.SparseCategoricalCrossentropy(from_logits=True)
cost(target, hypothesis)
tf.math.reduce_mean(-tf.math.log([0.95, 0.1]))                                                            # from_logits=False
tf.math.reduce_mean(-tf.math.log([tf.nn.softmax([0.05, 0.95, 0])[1], tf.nn.softmax([0.2, 0.7, 0.1])[2]])) # from_logits=True

target = [[0, 1], [0, 0]]
hypothesis = [[0.6, 0.4], [0.4, 0.6]]
cost = losses.CategoricalHinge()
cost(target, hypothesis) 

target = [[0., 1.], [1., 1.]]
hypothesis = [[1., 0.], [1., 1.]]
cost = losses.CosineSimilarity(axis=1)
cost(target, hypothesis) 

target = [[0., 1.], [0., 0.]]
hypothesis = [[0.6, 0.4], [0.4, 0.6]]
cost = losses.Hinge()
cost(target, hypothesis) 

target = [[0, 1], [0, 0]]
hypothesis = [[0.6, 0.4], [0.4, 0.6]]
cost = losses.Huber()
cost(target, hypothesis) 

target = [[0, 1], [0, 0]]
hypothesis = [[0.6, 0.4], [0.4, 0.6]]
cost = losses.KLDivergence()
cost(target, hypothesis) 

target = [[0., 1.], [0., 0.]]
hypothesis = [[1., 1.], [0., 0.]]
cost = losses.LogCosh()
cost(target, hypothesis) 

target = [[0., 1.], [0., 0.]]
hypothesis = [[1., 1.], [1., 0.]]
cost = losses.MeanAbsoluteError()
cost(target, hypothesis) 

target = [[2., 1.], [2., 3.]]
hypothesis = [[1., 1.], [1., 0.]]
cost = losses.MeanAbsolutePercentageError()
cost(target, hypothesis) 

target = [[0., 1.], [0., 0.]]
hypothesis = [[1., 1.], [1., 0.]]
cost = losses.MeanSquaredError()
cost(target, hypothesis) 

target = [[0., 1.], [0., 0.]]
hypothesis = [[1., 1.], [1., 0.]]
cost = losses.MeanSquaredLogarithmicError()
cost(target, hypothesis) 

target = [[0., 1.], [0., 0.]]
hypothesis = [[1., 1.], [0., 0.]]
cost = losses.Poisson()
cost(target, hypothesis) 

target = [[0., 1.], [0., 0.]]
hypothesis = [[0.6, 0.4], [0.4, 0.6]]
cost = losses.SquaredHinge()
cost(target, hypothesis) 
```

<br><br><br>


### Evaluation
- https://www.tensorflow.org/api_docs/python/tf/metrics
  
`Custom`
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
```python
import tensorflow as tf
from tensorflow.keras import metrics

y_true = [[1], [2], [3], [4]]
y_pred = [[0], [2], [3], [4]]
metric = metrics.Accuracy(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()

y_true = [[1], [1], [0], [0]]
y_pred = [[0.98], [1], [0], [0.6]]
metric = metrics.BinaryAccuracy(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.binary_accuracy(y_true, y_pred)

y_true = [0, 1, 1, 1]
y_pred = [1, 0, 1, 1]
metric = metrics.Precision(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()

y_true = [0, 1, 1, 1]
y_pred = [1, 0, 1, 1]
metric = metrics.Recall(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()

y_true = [[0, 1],[0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]
metric = metrics.BinaryCrossentropy(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.binary_crossentropy(y_true, y_pred)

y_true = [[0, 0, 1], [0, 1, 0]]
y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
metric = metrics.CategoricalAccuracy(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.categorical_accuracy(y_true, y_pred)

y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
metric = metrics.CategoricalCrossentropy(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.categorical_crossentropy(y_true, y_pred)

y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]
metric = metrics.CategoricalHinge(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()

y_true = [[0., 1.], [1., 1.]]
y_pred = [[1., 0.], [1., 1.]]
metric = metrics.CosineSimilarity(axis=1); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()

y_true = [0, 1, 1, 1]
y_pred = [0, 1, 0, 0]
metric = metrics.FalseNegatives(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()

y_true = [0, 1, 0, 0]
y_pred = [0, 0, 1, 1]
metric = metrics.FalsePositives(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()

y_true = [0, 1, 0, 0]
y_pred = [1, 1, 0, 0]
metric = metrics.TrueNegatives(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()

y_true = [0, 1, 1, 1]
y_pred = [1, 0, 1, 1]
metric = metrics.TruePositives(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()

y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]
metric = metrics.Hinge(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.hinge(y_true, y_pred)

y_true = [[0, 1], [0, 0]]
y_pred = [[0.6, 0.4], [0.4, 0.6]]
metric = metrics.KLDivergence(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.kl_divergence(y_true, y_pred)

y_true = [[0, 1], [0, 0]]
y_pred = [[1, 1], [0, 0]]
metric = metrics.LogCoshError(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()

y_pred = [1, 3, 5, 7]
metric = metrics.Mean(); metric.reset_state()
variable = metric.update_state(y_pred)
tensor = metric.result()

y_true = [[0, 1], [0, 0]]
y_pred = [[1, 1], [0, 0]]
metric = metrics.MeanAbsoluteError(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.mean_absolute_error(y_true, y_pred)

y_true = tf.constant([[0, 1], [0, 0]], dtype=tf.float32)
y_pred = tf.constant([[1, 1], [0, 0]], dtype=tf.float32)
metric = metrics.MeanAbsolutePercentageError(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.mean_absolute_percentage_error(y_true, y_pred)

y_true = tf.constant([[0, 1], [0, 0]], dtype=tf.float32)
y_pred = tf.constant([[1, 1], [0, 0]], dtype=tf.float32)
metric = metrics.MeanSquaredError(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.mean_squared_error(y_true, y_pred)

y_true = tf.constant([[0, 1], [0, 0]], dtype=tf.float32)
y_pred = tf.constant([[1, 1], [0, 0]], dtype=tf.float32)
metric = metrics.MeanSquaredLogarithmicError(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.mean_squared_logarithmic_error(y_true, y_pred)

y_true = [[2], [1]]
y_pred = [[0.1, 0.6, 0.3], [0.05, 0.95, 0]]
metric = metrics.SparseCategoricalAccuracy(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.sparse_categorical_accuracy(y_true, y_pred)

y_true = [1, 2]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
metric = metrics.SparseCategoricalCrossentropy(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.sparse_categorical_crossentropy(y_true, y_pred)

y_true = [2, 1]
y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
metric = metrics.SparseTopKCategoricalAccuracy(k=1); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
tensor = metrics.sparse_top_k_categorical_accuracy(y_true, y_pred)

y_true = [[0, 1], [0, 0]]
y_pred = [[1, 1], [0, 0]]
metric = metrics.RootMeanSquaredError(); metric.reset_state()
variable = metric.update_state(y_true, y_pred)
tensor = metric.result()
```
<br><br><br>

---


## Tensorboard
### Installation
```bash
$ pip install -U tensorboard-plugin-profile
```

### Logfile Generation
```python
import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Input(shape=(16,)))
model.add(layers.Dense(units=32))

# tensorboard
tensorboard_rootpath = "logs"
tensorboard_logpath = os.path.join(tensorboard_rootpath, datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(tensorboard_logpath, histogram_freq=1)
model.compile(optimizer="Adam", 
              loss="mse", 
              metrics=["mae"])
model.fit(callbacks=[tensorboard_callback], 
          x=tf.random.normal(shape=(10,16)), 
          y=tf.random.normal(shape=(10,32)), 
          validation_data=(tf.random.normal(shape=(10,16)), tf.random.normal(shape=(10,32))), 
          epochs=5)
```

### Execution
`jupyter mode`
```python
%load_ext tensorboard
%tensorboard --logdir tensorboard_rootpath
```

`browser mode`
```python
import subprocess

tensorboard_rootpath = 'logs'
server = subprocess.Popen(["tensorboard", "--logdir", f"{tensorboard_rootpath}", "--port=6006"])
server.kill()
```
```python
# logs: tensorboard_rootpath
!tensorboard --logdir logs --port=6006
```

![image](https://user-images.githubusercontent.com/70621679/150671892-b850b7aa-ea68-49d2-a149-28993a56f166.png)
![image](https://user-images.githubusercontent.com/70621679/150672132-eb21ffbc-029e-4464-8fb6-5f5846e2a158.png)


### Analysis
```python

```
