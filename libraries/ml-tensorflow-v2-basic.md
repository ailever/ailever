## [Deep Learning] | [tensorflow](https://www.tensorflow.org/api_docs/python/) | [github](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python)
- https://www.tensorflow.org/guide
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

c = tf.constant(1+3j, dtype=tf.dtypes.complex64)
tf.math.real(c)
tf.math.imag(c)

# einsum notation
m0 = tf.random.normal(shape=[2, 3])
m1 = tf.random.normal(shape=[3, 5])
tf.einsum('ij,jk->ik', m0, m1)
```

### Tensor Linear Algebra
```python
import tensorflow as tf

a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2])
tf.linalg.matmul(a, b)
```


<br><br><br>


---

## Module
- https://www.tensorflow.org/guide/intro_to_modules

### Module
```python
```

### Layer
```python
```

### Model
```python
```

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
import numpy as np

# Data
X = [1, 2, 3, 4, 5]
Y = [1, 2, 3, 4, 5]

# W, b initialize
W = tf.Variable(2.9)
b = tf.Variable(0.5)

# W, b update
learning_rate = 0.3
for i in range(300):
    # forward
    hypothesis = W * X
    cost = tf.reduce_mean(tf.square(hypothesis - Y))
    
    # backward
    W_grad = tf.reduce_mean(tf.multiply(tf.multiply(W, X) + b - Y, X))
    b_grad = tf.reduce_mean(tf.multiply(tf.multiply(W, X) + b - Y, 1))    
    W.assign(W - tf.multiply(learning_rate, W))
    b.assign(b - tf.multiply(learning_rate, b))
    
    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4f}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))

# predict
print(W.numpy(), b.numpy())
```
`with gradient tape`
```python
import tensorflow as tf
import numpy as np

# Data
X = [1, 2, 3, 4, 5]
Y = [1, 2, 3, 4, 5]

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
import numpy as np
import tensorflow as tf

data = np.array(   
     # X1,   X2,    X3,   y
    [[ 73.,  80.,  75., 152. ],
     [ 93.,  88.,  93., 185. ],
     [ 89.,  91.,  90., 180. ],
     [ 96.,  98., 100., 196. ],
     [ 73.,  66.,  70., 142. ]], dtype=np.float32)
X = data[:, :-1]
y = data[:, [-1]]

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
`logit`
```python
import tensorflow as tf 

def forward(feature, params:list):
    hypothesis = tf.divide(1., 1. + tf.exp(tf.matmul(feature, W) + b))
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
X = [[1., 2.],
     [2., 3.],
     [3., 1.],
     [4., 3.],
     [5., 3.],
     [6., 2.]]
Y = [[0.],
     [0.],
     [0.],
     [1.],
     [1.],
     [1.]]
dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(len(X))#.repeat()

for step in range(1001):
    for feature, target in iter(dataset):
        # forward
        with tf.GradientTape() as tape:
            hypothesis = forward(feature, params=[W, b])
            cost = loss_fn(hypothesis, target)

        # backward
        grads = tape.gradient(cost, [W, b])
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W,b]))
        if step % 100 == 0:
            print("Iter: {}, Loss: {:.4f}".format(step, loss(hypothesis, target)))
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

X = [[1., 2., 1., 1.],
     [2., 1., 3., 2.],
     [3., 1., 3., 4.],
     [4., 1., 5., 5.],
     [1., 7., 5., 5.],
     [1., 2., 5., 6.],
     [1., 6., 6., 6.],
     [1., 7., 7., 7.]]
Y = [[1., 0.],
     [1., 0.],
     [1., 0.],
     [1., 0.],
     [0., 1.],
     [0., 1.],
     [0., 1.],
     [0., 1.]]

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

X = [[1., 2., 1., 1.],
     [2., 1., 3., 2.],
     [3., 1., 3., 4.],
     [4., 1., 5., 5.],
     [1., 7., 5., 5.],
     [1., 2., 5., 6.],
     [1., 6., 6., 6.],
     [1., 7., 7., 7.]]
Y = [[0., 0., 1.],
     [0., 0., 1.],
     [0., 0., 1.],
     [0., 1., 0.],
     [0., 1., 0.],
     [0., 1., 0.],
     [1., 0., 0.],
     [1., 0., 0.]]

model = Architecture()
model.fit(X, Y)
```


### Multi-Label Classification
```python
```

## Convolutnal Neural Network


## Recurrent Neural Network

<br><br><br>

---

## API

### Layer
- https://www.tensorflow.org/api_docs/python/tf/keras/layers
- https://www.tensorflow.org/api_docs/python/tf/keras/activations
- https://www.tensorflow.org/api_docs/python/tf/keras/initializers
- https://www.tensorflow.org/api_docs/python/tf/keras/Model
- https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
- https://www.tensorflow.org/api_docs/python/tf/keras/Input
```python
```


<br><br><br>

### Cost Function
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
tf.math.reduce_mean(-tf.math.log([(1-0.3), 0.8, 0.5, 0.2]))

target = [[0, 1, 0], [0, 0, 1]]
hypothesis = [[0.05, 0.95, 0], [0.2, 0.7, 0.1]]
cost = losses.CategoricalCrossentropy()
print(cost(target, hypothesis) .numpy())
tf.math.reduce_mean(-tf.math.log([0.95, 0.1]))

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

target = [1, 2]
hypothesis = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
cost = losses.SparseCategoricalCrossentropy()
cost(target, hypothesis) 

target = [[0., 1.], [0., 0.]]
hypothesis = [[0.6, 0.4], [0.4, 0.6]]
cost = losses.SquaredHinge()
cost(target, hypothesis) 
```

<br><br><br>


### Evaluation
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



