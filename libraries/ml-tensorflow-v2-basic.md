## [Deep Learning] | [tensorflow](https://www.tensorflow.org/api_docs/python/) | [github](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python)
### Navigation
- https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#installation
- https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#tensorflow-basic
- https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#tensorflow-advanced
    - https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#computational-graph
- https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#module
    - https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#layer
    - https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#model
        - https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#sequential-api-model
        - https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#functional-api-model
        - https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#subclassing-api-model
    - https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#train
    - https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#callbacks
- https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#keras-api
    - https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#layer-1
    - https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#optimizer-1
    - https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#cost-function
    - https://github.com/ailever/ailever/blob/master/libraries/ml-tensorflow-v2-basic.md#evaluation

## Installation
```bash
$ pip install tensorflow
$ pip install tensorflow_datasets
$ pip install pydot
```

---

<br><br><br>

---

## TensorFlow Basic
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


## Tensorflow Advanced
### Gradient
- https://www.tensorflow.org/guide/autodiff
- https://www.tensorflow.org/guide/advanced_autodiff

#### First-order Gradient
```python
# [tape.watch]
import tensorflow as tf

w1 = tf.constant([2.0], name='W1Tensor')
w2 = tf.Variable([2.0], trainable=True, name='W2Tensor')
with tf.GradientTape() as tape:
    tape.watch(w1)         # watching for gradient of constant tensor
    cost1 = tf.exp(w1 * w1)
    cost2 = tf.exp(w2 * w2)

[var.name for var in tape.watched_variables()]
```

```python
# [constant vs Variable]
import tensorflow as tf

w1 = tf.constant([2.])
w2 = tf.Variable([2.], trainable=True)
with tf.GradientTape() as tape:
    cost = 2*w1*w2 + 3*w2**2

gradients = tape.gradient(cost, {'w1':w1, 'w2':w2})
gradients['w1'], gradients['w2']
```
```
(None,
 <tf.Tensor: shape=(1,), dtype=float32, numpy=array([16.], dtype=float32)>)
```


`Constant Gradient`
```python
# [tape.watched_variables] : Not Applicable for constant tensor 
# [w.assign_sub] : Not Applicable for constant tensor (error)
import tensorflow as tf

w = tf.constant([2.0], name='WTensor')
with tf.GradientTape() as tape:
    tape.watch(w)         # watching for gradient of constant tensor
    cost = tf.exp(w * w)  # exp(w**2)*(2w) => 218.3926

gradients = tape.gradient(cost, {'w': w})
watched_variables = [variable for variable in tape.watched_variables()]
del tape

print('[d(cost)/d(w)]:', gradients['w'])  
# w.assign_sub(0.01*gradients['w']) # Not Applicable
```

`Variable Gradient`
```python
# [tf.stop_gradient]
import tensorflow as tf

w1 = tf.Variable([2.], trainable=True, name='W1Tensor')
w2 = tf.Variable([2.], trainable=True, name='W2Tensor')
with tf.GradientTape() as tape:
    cost = tf.stop_gradient(2*w1*w2) + 3*w2**2

gradients = tape.gradient(cost, {'w1':w1, 'w2':w2})
gradients['w1'], gradients['w2']
```
```
(None,
 <tf.Tensor: shape=(1,), dtype=float32, numpy=array([12.], dtype=float32)>)
```

```python
# [tape.stop_recording]
import tensorflow as tf

w1 = tf.Variable([2.], trainable=True, name='W1Tensor')
w2 = tf.Variable([2.], trainable=True, name='W2Tensor')
with tf.GradientTape() as tape:
    with tape.stop_recording():
        out1 = 2*w1*w2
    out2 = 3*w2**2
    cost = out1 + out2

gradients = tape.gradient(cost, {'w1': w1, 'w2': w2})
gradients['w1'], gradients['w2']
```
```
(None,
 <tf.Tensor: shape=(1,), dtype=float32, numpy=array([12.], dtype=float32)>)
```
```python
# [tape.reset]
import tensorflow as tf

w1 = tf.Variable([2.], trainable=True, name='W1Tensor')
w2 = tf.Variable([2.], trainable=True, name='W2Tensor')
with tf.GradientTape() as tape:
    out1 = 2*w1*w2
    tape.reset()
    out2 = 3*w2**2
    cost = out1 + out2

gradients = tape.gradient(cost, {'w1': w1, 'w2': w2})
gradients['w1'], gradients['w2']
```
```
(None,
 <tf.Tensor: shape=(1,), dtype=float32, numpy=array([12.], dtype=float32)>)
```

```python
# [watch_accessed_variables=False]
import tensorflow as tf

w = tf.Variable([2.0], trainable=True, name='WTensor')  # w.trainable = True
with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(w)
    cost = tf.exp(w * w)

gradients = tape.gradient(cost, {'w': w})
watched_variables = [variable for variable in tape.watched_variables()]
del tape

print('[d(cost)/d(w)]:', gradients['w'])  
w.assign_sub(0.01*gradients['w']) # Not Applicable
```

```python
# [(default) watch_accessed_variables=True]
import tensorflow as tf

w1 = tf.Variable([2.0], trainable=True, name='W1Tensor')  # w1.trainable = True
w2 = tf.Variable([2.0], trainable=False, name='W2Tensor') # w2.trainable = False
with tf.GradientTape() as tape:
    out1 = w1 * w1
    out2 = w2 * w2
    cost = out1 + out2

gradients = tape.gradient(cost, {'w1': w1, 'w2': w2})
watched_variables = [variable for variable in tape.watched_variables()]
del tape

print('[d(cost)/d(w1)]:', gradients['w1'])  # 2*x => 4
print('[d(cost)/d(w2)]:', gradients['w2'])  # None
w1.assign_sub(0.01*gradients['w1'])
w2.assign_sub(0.01*4)
```

```python
# [trainable=True/False]
import tensorflow as tf

w1 = tf.Variable([2.0], trainable=True, name='W1Tensor')
w2 = tf.Variable([2.0], trainable=False, name='W2Tensor')
with tf.GradientTape() as tape:
    out1 = w1 * w1
    out2 = w2 * w2
    cost = out1 + out2

gradients = tape.gradient(cost, [w1, w2])
watched_variables = [variable for variable in tape.watched_variables()]
del tape

print('[d(cost)/d(w1)]:', gradients[0])  # 2*(w1) => 4
print('[d(cost)/d(w2)]:', gradients[1])  # None
w1.assign_sub(0.01*gradients[0])
w2.assign_sub(0.01*4)
```


#### Second-order Gradient
```python
# [Second-Order Gradient]
import tensorflow as tf

w = tf.Variable([1.0])  # Create a Tensorflow variable initialized to 1.0
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = w**3

    # Compute the gradient inside the outer `tape2` context manager
    # which means the gradient computation is differentiable as well.
    gradient1 = tape1.gradient(y, w) # dy/dw
gradient2 = tape2.gradient(dy_dw, w) # d2y/dw2
gradient1, gradient2
```

#### Scalar Gradient

```python
# [persistent=True]
import tensorflow as tf

w = tf.Variable([2.0])
with tf.GradientTape(persistent=True) as tape:
    y0 = w**2
    y1 = 1 / w

print(tape.gradient(y0, w).numpy())
print(tape.gradient(y1, w).numpy())
print(tape.gradient([y0, y1], w).numpy())
```
```
[4.0]
[-0.25]
[3.75]
```

```python
# [(default) persistent=False]
import tensorflow as tf

w = tf.Variable([2.0])
with tf.GradientTape(persistent=False) as tape:
    y0 = w**2
    y1 = 1 / w

print(tape.gradient([y0, y1], w).numpy())
```
```
[3.75]
```

```python
import tensorflow as tf

w = tf.Variable([2.])
with tf.GradientTape() as tape:
    y = w * [3., 4.]
    
print(tape.gradient(y, w).numpy())
```
```
[7.]
```

#### Vector Gradient
`Scalar Input`
```python
import tensorflow

W = tf.linspace(-10.0, 10.0, 10+1) # [-10.,  -8.,  -6.,  -4.,  -2.,   0.,   2.,   4.,   6.,   8.,  10.]
with tf.GradientTape() as tape:
    tape.watch(W)
    Y = tf.nn.sigmoid(W)

tape.gradient(Y, W)
```
```
<tf.Tensor: shape=(11,), dtype=float32, numpy=
array([4.53978682e-05, 3.35312623e-04, 2.46652518e-03, 1.76627338e-02,
       1.04993574e-01, 2.50000000e-01, 1.04993574e-01, 1.76627338e-02,
       2.46652518e-03, 3.35342396e-04, 4.54166766e-05], dtype=float32)>    
```
```python
# [jacobian]
import tensorflow

W = tf.linspace(-10.0, 10.0, 10+1) # [-10.,  -8.,  -6.,  -4.,  -2.,   0.,   2.,   4.,   6.,   8.,  10.]
with tf.GradientTape(persistent=True) as tape:
    tape.watch(W)
    Y = tf.nn.sigmoid(W)

gradients = tape.gradient(Y, W) # (11, )
jacobian = tape.jacobian(Y, W)  # (11, 11)
tf.einsum('ij->j', jacobian) - gradients
```

`Vector Input`
```python
import tensorflow as tf

X = tf.constant(tf.random.normal([7, 5]))
W = tf.Variable(tf.random.normal(shape=(5, 10)))
with tf.GradientTape() as tape:
    Y = X@W
    
tape.gradient(Y, W)
```
```
* Gradient
<tf.Tensor: shape=(5, 10), dtype=float32, numpy=
array([[1.3293178, 1.3293178, 1.3293178, 1.3293178, 1.3293178, 1.3293178, 1.3293178, 1.3293178, 1.3293178, 1.3293178],  # X[:, 0].sum()
       [1.5503446, 1.5503446, 1.5503446, 1.5503446, 1.5503446, 1.5503446, 1.5503446, 1.5503446, 1.5503446, 1.5503446],  # X[:, 1].sum()
       [2.6367357, 2.6367357, 2.6367357, 2.6367357, 2.6367357, 2.6367357, 2.6367357, 2.6367357, 2.6367357, 2.6367357],  # X[:, 2].sum()
       [3.1581473, 3.1581473, 3.1581473, 3.1581473, 3.1581473, 3.1581473, 3.1581473, 3.1581473, 3.1581473, 3.1581473],  # X[:, 3].sum()
       [1.3615382, 1.3615382, 1.3615382, 1.3615382, 1.3615382, 1.3615382, 1.3615382, 1.3615382, 1.3615382, 1.3615382]], # X[:, 4].sum()
       dtype=float32)>

* X
<tf.Tensor: shape=(7, 5), dtype=float32, numpy=
array([[ 1.3878208 , -1.2954499 ,  1.2458328 ,  0.8125885 ,  1.3659256 ],
       [ 0.4081827 , -0.49392286,  0.66081274,  0.8159353 ,  1.1641886 ],
       [-1.7174034 ,  0.02244303,  1.8436242 ,  1.2765765 , -0.30611107],
       [ 0.2698866 ,  0.2977228 , -1.0660497 ,  0.67187244,  0.62355465],
       [-0.05231808,  0.31540915,  0.1730449 , -0.04806426,  0.26622197],
       [ 0.20742404,  0.5112282 , -0.7724665 ,  0.534478  , -0.7211987 ],
       [ 0.825725  ,  2.192914  ,  0.55193686, -0.905239  , -1.0310428 ]],
      dtype=float32)>

* W
<tf.Variable 'Variable:0' shape=(5, 10) dtype=float32, numpy=
array([[-1.0677733 , -0.7274096 , -0.1864897 ,  1.3681251 ,  2.1630728 , -0.7455596 , -0.30695024, -1.5998704 ,  1.9329438 ,  2.2270694 ],
       [ 1.8150767 , -1.2310882 ,  0.9339469 , -0.5144172 ,  0.77093524, -1.2204999 ,  0.17778336,  0.3722942 , -0.46277007, -0.00268906],
       [ 1.7084551 ,  0.8196859 , -1.1261114 ,  0.44070497,  0.31172493,  1.5580393 ,  0.24851267,  0.80291706,  0.3739408 ,  0.93399704],
       [-1.1445469 , -0.84283197, -1.961605  ,  0.0674562 , -1.1750969 , -2.042662  , -1.1516502 , -0.05241072,  1.2980733 , -0.8297142 ],
       [-0.1420171 ,  0.01554095,  2.1261415 ,  0.8713996 ,  0.81936836,  0.2917507 ,  0.06435041,  0.0905281 ,  1.757147  , -0.08798544]],
      dtype=float32)>
```
```python
# [jacobian(1)]
import tensorflow as tf

X = tf.constant(tf.random.normal([7, 5]))
W = tf.Variable(tf.random.normal(shape=(5, 10)))
with tf.GradientTape(persistent=True) as tape:
    Y = X@W
    
gradients = tape.gradient(Y, W) # (5, 10)   
jacobian = tape.jacobian(Y, W)  # (7, 10, 5, 10)
tf.einsum('ijkl->kl', jacobian) - gradients
```
```python
# [jacobian(2)]
import tensorflow as tf

X = tf.constant(tf.random.normal([7, 5]))
W = tf.Variable(tf.random.normal(shape=(5, 10)))
with tf.GradientTape(persistent=True) as tape:
    Y = X@W
    cost = tf.reduce_mean(Y**2)

gradients = tape.gradient(cost, W) # (5, 10)   
jacobian = tape.jacobian(cost, W)  # (5, 10)
jacobian - gradients
```
```python
# [hessian]
import tensorflow as tf

X = tf.constant(tf.random.normal([7, 5]))
W = tf.Variable(tf.random.normal(shape=(5, 10)))
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        Y = X@W
        cost = tf.reduce_mean(Y**2)
    gradients = tape1.gradient(cost, W)
hessian = tape2.jacobian(gradients, W) # (5, 10, 5, 10)
tf.einsum('ijkl->kl', hessian)         # (5, 10) 
```





#### Custom Gradient
`Custom Gradient`
```python
# Tensorflow Gradient Calculation
import tensorflow as tf

def Function(x):
    return tf.math.log(1 + tf.exp(x))

def Gradient(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        value = Function(x)
    return tape.gradient(value, x)

Gradient(tf.constant(0.)).numpy()   # 0.5 (numerically stable)
Gradient(tf.constant(100.)).numpy() # nan (numerically unstable)
```
```python
# Custom Gradient Definition
import tensorflow as tf

@tf.custom_gradient
def Function(x):
    def CustomGradient(dx):
        return dx * (1 - 1 / (1 + tf.exp(x)))
    return tf.math.log(1 + tf.exp(x)), CustomGradient

def Gradient(x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        value = Function(x)
    return tape.gradient(value, x)

Gradient(tf.constant(0.)).numpy()   # 0.5
Gradient(tf.constant(100.)).numpy() # 1.0
```

#### Optimizer
```python
import tensorflow as tf
from tensorflow.keras import optimizers

@tf.function
def Function(x):
    return W - x

W = tf.Variable([2.])
x = tf.constant([1.])
optimizer = optimizers.Adam(0.1)
with tf.GradientTape() as tape:
    y = Function(x)
    
gradient = tape.gradient(y, [W])
optimizer.apply_gradients(zip(gradient, [W]))
W.numpy()
```

```python
import tensorflow as tf
from tensorflow.keras import optimizers

W = tf.Variable(tf.ones(shape=(10,4)), trainable=True)
Y = tf.constant(tf.zeros(shape=(10,4)))

trainable_list_fn = lambda : W
loss_fn = lambda: tf.reduce_mean((W - Y)**2)

lr_schedule = 0.5
#optimizer = optimizers.SGD(learning_rate=lr_schedule, momentum=0.0, nesterov=False, name='SGD')
#optimizer = optimizers.RMSprop(learning_rate=lr_schedule, rho=0.9, momentum=0.0, epsilon=1e-07, centered=False, name='RMSprop')
#optimizer = optimizers.Adagrad(learning_rate=lr_schedule, initial_accumulator_value=0.1, epsilon=1e-07, name='Adagrad')
#optimizer = optimizers.Adadelta(learning_rate=lr_schedule, rho=0.95, epsilon=1e-07, name='Adadelta')
#optimizer = optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
optimizer = optimizers.Adamax(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax')
#optimizer = optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
#optimizer = optimizers.Ftrl(learning_rate=0.001, learning_rate_power=-0.5, initial_accumulator_value=0.1, l1_regularization_strength=0.0, l2_regularization_strength=0.0, name='Ftrl', l2_shrinkage_regularization_strength=0.0, beta=0.0)

for _ in range(100):
    optimizer.minimize(loss_fn, trainable_list_fn)

W.numpy()
```

### Computational Graph
- https://www.tensorflow.org/guide/intro_to_graphs
- https://www.tensorflow.org/guide/function
- https://www.tensorflow.org/guide/graph_optimization

`tf.function`
```python
import tensorflow as tf

@tf.function
def forward(x, y):
    return (x-y)**2

W = tf.Variable([2.0])
with tf.GradientTape() as tape:
    result = forward(W, tf.constant([1.0]))
tape.gradient(result, W)
```

#### Graph Execution
```python
import tensorflow as tf

tf.config.run_functions_eagerly(False)

@tf.function
def forward(x, y):
    mse = (x-y)**2
    print('- python print', ':', mse)
    tf.print('%-14s'%'- tf print', ':', mse)

forward(tf.constant([1., 2.]), tf.constant([1., 2.]))    
forward(tf.constant([1., 2.]), tf.constant([1., 2.]))    
forward(tf.constant([1., 2.]), tf.constant([1., 2.]))
```
```python
import tensorflow as tf
import timeit

tf.config.run_functions_eagerly(False)

def eager_measure(x, steps):
    result = tf.eye(10, dtype=tf.dtypes.int32)
    for _ in range(steps):
        result = tf.matmul(x, result)
    return result

@tf.function
def graph_measure(x, steps):
    result = tf.eye(10, dtype=tf.dtypes.int32)
    for _ in range(steps):
        result = tf.matmul(x, result)
    return result
#graph_measure = tf.function(eager_measure)

x = tf.random.uniform(shape=[10, 10], minval=-1, maxval=2, dtype=tf.dtypes.int32)
print('EAGER: ', timeit.timeit(lambda : eager_measure(x, 200), number=1000))
print('GRAPH: ', timeit.timeit(lambda : graph_measure(x, 200), number=1000))
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

```python
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics
from sklearn.model_selection import train_test_split

class Model(tf.Module):
    def __init__(self, name=None):
        super(Model, self).__init__(name)
        self.W1 = tf.Variable(tf.ones(shape=[10, 8]), trainable=True, name="kernel1")
        self.b1 = tf.Variable(tf.ones(shape=[1 , 8]), trainable=True, name="bias1")
        self.W2 = tf.Variable(tf.ones(shape=[8 , 4]), trainable=True, name="kernel2")
        self.b2 = tf.Variable(tf.ones(shape=[1 , 4]), trainable=True, name="bias2")
        self.W3 = tf.Variable(tf.ones(shape=[4 , 2]), trainable=True, name="kernel3")
        self.b3 = tf.Variable(tf.ones(shape=[1 , 2]), trainable=True, name="bias3")

    def __call__(self, x, training=False):
        x = tf.nn.relu(x@self.W1 + self.b1)
        x = tf.nn.relu(x@self.W2 + self.b2)
        x = tf.nn.sigmoid(x@self.W3 + self.b3)
        return x

# [Dataset]
X = tf.random.normal(shape=(1000, 10))
y = tf.random.normal(shape=(1000, 2))
train_X, test_X, train_y, test_y = train_test_split(X.numpy(), y.numpy(), test_size=0.3, shuffle=False)
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).repeat(10).batch(10)
test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y)).batch(1)

# [Modeling Instances]
model = Model()
criterion = losses.MeanSquaredError()
optimizer = optimizers.Adam(0.1)
train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')

# [Train Step]
@tf.function
def train_step(features, targets):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        cost = criterion(targets, predictions)
    gradients = tape.gradient(cost, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(cost)
    
# [Test Step]
@tf.function
def test_step(features, targets):
    predictions = model(features, training=False)
    cost = criterion(targets, predictions)
    test_loss(cost)
    
EPOCHS = 5
saving_paths = list()
for epoch in range(EPOCHS):
    train_loss.reset_states()
    test_loss.reset_states()

    for features, targets in train_dataset:
        train_step(features, targets)
            
    for features, targets in test_dataset:
        test_step(features, targets)

    print(
        f'[Epoch {epoch + 1}], '
        f'Train Loss: {train_loss.result()}, '
        f'Test Loss: {test_loss.result()}, '
    )


tf.saved_model.save(model, "model/version/1/")
model = tf.saved_model.load("model/version/1/")
```
```
$ [./model/version/1/]   assets  saved_model.pb  variables

- assets         : tensorflow graph text files
- saved_model.pb : function set with signitures
- variables      : train checkpoints
  - variables.data-00000-of-00001
  - variables.index
```

### Layer
#### Custom Layer Definition
```python
import tensorflow as tf
from tensorflow.keras import layers

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
        return {"units": self.units, "name": self.name}

layer = CustomLayer(units=32, name='custom_layer')
layer.build(input_shape=(1000, 64))

config = layers.serialize(layer)
config_details = layer.get_weights()

layer = layers.deserialize(config, custom_objects={"CustomLayer": CustomLayer})
layer = CustomLayer.from_config(config['config'])
```

```python
import tensorflow as tf
from tensorflow.keras import layers

class CustomLayer(layers.Layer):
    def __init__(self, units=32, trainable=True, dtype=tf.float32, name=None):
        super(CustomLayer, self).__init__(trainable=trainable, dtype=dtype, name=name)
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
        config = super(CustomLayer, self).get_config()
        config.update({"units": self.units})
        return config 

layer = CustomLayer(units=32, name='custom_layer')
layer.build(input_shape=(1000, 64))

config = layers.serialize(layer)
config_details = layer.get_weights()

layer = layers.deserialize(config, custom_objects={"CustomLayer": CustomLayer})
layer = CustomLayer.from_config(config['config'])
```

#### Custom Layer Checkpoint
```python
import tensorflow as tf
from tensorflow.keras import layers

class CustomLayer(layers.Layer):
    def __init__(self, units=32, trainable=True, dtype=tf.float32, name=None):
        super(CustomLayer, self).__init__(trainable=trainable, dtype=dtype, name=name)
        self.weight = tf.Variable(tf.zeros(shape=(100, units)), name="weight")

layer = CustomLayer(units=32)

save_path = tf.train.Checkpoint(layer=layer).save("variables") # variables-1.index, variables-1.data-00000-of-00001, 
ckpt_reader = tf.train.load_checkpoint(save_path)              # variables-1
ckpt_reader.get_variable_to_dtype_map()
```

#### Custom Layer Usage
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
history = model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,10)))
history.params
history.history

# Functional Model
inputs = layers.Input((100,))
outputs = CustomLayer(10)(inputs)
model = models.Model(inputs, outputs, name='CustomModel')
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
history = model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,10)))
history.params
history.history

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
- https://www.tensorflow.org/guide/checkpoint
- https://www.tensorflow.org/guide/saved_model
- https://www.tensorflow.org/guide/keras/save_and_serialize
- https://www.tensorflow.org/tutorials/keras/save_and_load
- https://www.tensorflow.org/api_docs/python/tf/train/load_checkpoint
- https://keras.io/guides/serialization_and_saving/
- https://keras.io/api/models/model_saving_apis/

`Layer` vs `Model`
- model.fit(), model.evaluate(), model.predict()
- save and serialization API(save(), save_weights()...)

#### Sequential API Model
##### Sequential Model Definition
`with built-in layers`
```python
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Dense(4, name='1L', activation="relu"))
model.add(layers.Dense(4, name='2L', activation="relu"))
model.add(layers.Dense(4, name='3L'))
model.build(input_shape=(1000,128))

config = model.get_config()
config_details = model.get_weights()

model = models.Sequential.from_config(config)
model.set_weights(config_details)

model = models.clone_model(model)
model.set_weights(config_details)
```

`with custom layer`
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
model.add(CustomLayer(32, name='CustomLayer'))
model.build(input_shape=(1000,128))

config = model.get_config()
config_details = model.get_weights()
custom_objects = {"CustomLayer": CustomLayer}
with tf.keras.utils.custom_object_scope(custom_objects):
    model = models.Sequential.from_config(config)
    model.set_weights(config_details)
    
    model = models.clone_model(model)
    model.set_weights(config_details)
```

##### Sequential Model Checkpoint

`save_weights`
```python
import tensorflow as tf
from tensorflow.keras import models, layers

model = models.Sequential(
    [
        layers.Input(shape=(784,), name="digits"),
        layers.Dense(64, activation="relu", name="dense_1"),
        layers.Dense(64, activation="relu", name="dense_2"),
        layers.Dense(10, name="predictions"),
    ]
)
model.save_weights("sequential") # sequential.index, sequential.data-00001-of-00001
ckpt_reader = tf.train.load_checkpoint('sequential')
ckpt_reader.get_variable_to_dtype_map()
```
```
{'layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE': tf.float32,
 'layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE': tf.float32,
 '_CHECKPOINTABLE_OBJECT_GRAPH': tf.string,
 'layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE': tf.float32,
 'layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE': tf.float32,
 'layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE': tf.float32,
 'layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE': tf.float32}
```

`Checkpoint Object`
```python
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics

model = models.Sequential()
model.add(layers.Dense(4, name='L1'))

optimizer = optimizers.Adam(0.1)
mse_loss = losses.MeanSquaredError()
mae_metric = metrics.MeanAbsoluteError()
model.compile(optimizer=optimizer, loss=mse_loss, metrics=[mae_metric])
model.fit(tf.random.normal(shape=(100,10)), tf.random.normal(shape=(100,4)), verbose=0, epochs=10)

saving_ckpt = tf.train.Checkpoint(model=model)
save_path = saving_ckpt.save('sequential')
ckpt_reader = tf.train.load_checkpoint(save_path)
ckpt_reader.get_variable_to_dtype_map() # tf.train.list_variables('./')
```
```
[('_CHECKPOINTABLE_OBJECT_GRAPH', []),
 ('model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE', [4]),
 ('model/layer_with_weights-0/bias/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE',
  [4]),
 ('model/layer_with_weights-0/bias/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE',
  [4]),
 ('model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE', [10, 4]),
 ('model/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE',
  [10, 4]),
 ('model/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE',
  [10, 4]),
 ('model/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('model/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('model/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('model/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('model/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('save_counter/.ATTRIBUTES/VARIABLE_VALUE', [])]
```


`model tracking`
```python
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics

model = models.Sequential()
model.add(layers.Dense(4, name='L1'))

optimizer = optimizers.Adam(0.1)
mse_loss = losses.MeanSquaredError()
mae_metric = metrics.MeanAbsoluteError()
model.compile(optimizer=optimizer, loss=mse_loss, metrics=[mae_metric])
model.fit(tf.random.normal(shape=(100,10)), tf.random.normal(shape=(100,4)), verbose=0, epochs=10)

# saving checkpoint
saving_ckpt = tf.train.Checkpoint(model=model)
save_path = saving_ckpt.save('sequential')

# tracking(loading weights)
class Tracking(models.Model):
    def __init__(self):
        super(Tracking, self).__init__()
        self.layer = model.layers[0] # tracking target

# tracking checkpoint
tracking = Tracking()
tracking_ckpt = tf.train.Checkpoint(target=tracking)
save_path = tracking_ckpt.save('tracking')
tracking_ckpt_reader = tf.train.load_checkpoint(save_path)
tf.print(tf.train.list_variables('./')) #ckpt_reader.get_variable_to_dtype_map()

# define computational graph with model architecture
bias = tf.Variable(tf.zeros(shape=(4,)))
kernel = tf.Variable(tf.zeros(shape=(10, 4)))
layer = tf.train.Checkpoint(bias=bias, kernel=kernel)
target = tf.train.Checkpoint(layer=layer)
root = tf.train.Checkpoint(target=target)
root.restore(tf.train.latest_checkpoint('./'))
print(bias.numpy())
print(kernel.numpy())
```

##### Sequential Model Training
`training flow`
```python
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics, callbacks, utils

model = models.Sequential()
model.add(layers.Dense(4, name='1L', activation="relu"))
model.add(layers.Dense(4, name='2L', activation="relu"))
model.add(layers.Dense(4, name='3L'))

# training instances
optimizer = optimizers.Adam(0.1)
mse_loss = losses.MeanSquaredError()
mae_metric = metrics.MeanAbsoluteError()
cb_lambda = callbacks.LambdaCallback(
    on_epoch_begin = lambda epoch, logs: print(f'[on_epoch_begin]: {epoch}, {logs}'),
    on_epoch_end   = lambda epoch, logs: print(f"[on_epoch_end]: {epoch}, {logs}"),
    on_batch_begin = lambda batch, logs: print(f"[on_batch_begin]: {batch}, {logs}"),
    on_batch_end   = lambda batch, logs: print(f"[on_batch_end]: {batch}, {logs}"),
    on_train_begin = lambda        logs: print(f"[on_train_begin]: {logs}"),
    on_train_end   = lambda        logs: print(f"[on_train_end]: {logs}"),)

# training
model.compile(optimizer=optimizer, loss=mse_loss, metrics=[mae_metric])
model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,4)), epochs=10, callbacks=[cb_lambda])
model.predict(tf.random.normal(shape=(1,100)))  # return prediction result
model.evaluate(tf.random.normal(shape=(1,100))) # return loss and metric 

tf.keras.backend.clear_session()
model.save("model/version/1")                            # creates a assets(folder), saved_model.pb(file), variables(folder) 
model.save_weights('model/version/1/variables/weights')  # creates a checkpoint, weights.data*, weights.index on the variables(folder)
model = models.load_model("model/version/1")
model.load_weights('model/version/1/variables/weights')  # check "tf.train.latest_checkpoint('model/version/1/variables')"
model.summary()

utils.plot_model(model, "model.png", show_shapes=True)
```

`training details`
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
model.layers

# model configuration& architecture saving
config = model.get_config()
model = models.Sequential.from_config(config)

# training
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.compiled_metrics
model.compiled_loss
model.optimizer
model.optimizer.get_config()
model.loss

history = model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,4)))
history.params
history.history
"""
history = model.fit(x=None, y=None, batch_size=None, shuffle=True, callbacks=None, 
    epochs=1, initial_epoch=0, verbose='auto', steps_per_epoch=None, class_weight=None, sample_weight=None, 
    validation_split=0.0, validation_data=None, validation_batch_size=None, validation_steps=None, validation_freq=1,
    max_queue_size=10, workers=1, use_multiprocessing=False)"""
model.compiled_metrics.metrics
model.compiled_loss.metrics
model.optimizer.get_weights()
model.inputs
model.outputs
model.metrics

# prediction
model.predict(tf.random.normal(shape=(1,100)))  # return prediction result

# evaluation
model.evaluate(tf.random.normal(shape=(1,100))) # return loss and metric 

# save & load (1: full model): Keras H5 format
model.save('model.h5')                 # creates a HDF5 file 'model.h5'
model = models.load_model('model.h5')
model.save_weights("weights.h5")       # creates a HDF5 file 'weights.h5'
model.load_weights("weights.h5")
model.optimizer.get_config()
model.optimizer.get_weights()
#model.add_loss(~)
#model.add_metric(~)

# save & load (2: full model): SavedModel format
model.save("model/version/1")                            # creates a assets(folder), saved_model.pb(file), variables(folder) 
model = models.load_model("model/version/1")
model.save_weights('model/version/1/variables/weights')  # creates a checkpoint, weights.data*, weights.index on the variables(folder)
model.load_weights('model/version/1/variables/weights')  # check "tf.train.latest_checkpoint('model/version/1/variables')"
model.optimizer.get_config()
model.optimizer.get_weights()

# model entities
model.submodules
model.submodules[-1].input
model.submodules[-1].output
model.variables
model.trainable_variables
model.non_trainable_variables

model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
```

#### Functional API Model
##### Functional Model Definition
`with bulit-in layers`
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

# model configuration& architecture saving
config = model.get_config()
config_details = model.get_weights()

model = models.Model.from_config(config)
model.set_weights(config_details)

model = models.clone_model(model)
model.set_weights(config_details)
```
`with custom layer`
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

# Functional Model
inputs = layers.Input((100,))
outputs = CustomLayer(10)(inputs)
model = models.Model(inputs, outputs, name='CustomModel')

config = model.get_config()
config_details = model.get_weights()
custom_objects = {"CustomLayer": CustomLayer}
with tf.keras.utils.custom_object_scope(custom_objects):
    model = models.Model.from_config(config)
    model.set_weights(config_details)    
    
    model = models.clone_model(model)
    model.set_weights(config_details)
```

##### Functional Model Checkpoint
`save_weights`
```python
import tensorflow as tf
from tensorflow.keras import models, layers

inputs = layers.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, name="predictions")(x)
model = models.Model(inputs, outputs)

model.save_weights("functional") # functional.index, functional.data-00001-of-00001
ckpt_reader = tf.train.load_checkpoint('functional')
ckpt_reader.get_variable_to_dtype_map()
```


`Checkpoint Object`
```python
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics

inputs = layers.Input(shape=(None, 10))
outputs = layers.Dense(4, name='L1')(inputs)
model = models.Model(inputs, outputs)

optimizer = optimizers.Adam(0.1)
mse_loss = losses.MeanSquaredError()
mae_metric = metrics.MeanAbsoluteError()
model.compile(optimizer=optimizer, loss=mse_loss, metrics=[mae_metric])
model.fit(tf.random.normal(shape=(100,10)), tf.random.normal(shape=(100,4)), verbose=0, epochs=10)

saving_ckpt = tf.train.Checkpoint(model=model)
save_path = saving_ckpt.save('functional')
ckpt_reader = tf.train.load_checkpoint(save_path)
ckpt_reader.get_variable_to_dtype_map() # tf.train.list_variables('./')
```
```
[('_CHECKPOINTABLE_OBJECT_GRAPH', []),
 ('model/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE', [4]),
 ('model/layer_with_weights-0/bias/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE',
  [4]),
 ('model/layer_with_weights-0/bias/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE',
  [4]),
 ('model/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE', [10, 4]),
 ('model/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE',
  [10, 4]),
 ('model/layer_with_weights-0/kernel/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE',
  [10, 4]),
 ('model/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('model/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('model/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('model/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('model/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('save_counter/.ATTRIBUTES/VARIABLE_VALUE', [])]
```

`model tracking`
```python
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics

inputs = layers.Input(shape=(None, 10))
outputs = layers.Dense(4, name='1L', activation="relu")(inputs)
model = models.Model(inputs, outputs)

optimizer = optimizers.Adam(0.1)
mse_loss = losses.MeanSquaredError()
mae_metric = metrics.MeanAbsoluteError()
model.compile(optimizer=optimizer, loss=mse_loss, metrics=[mae_metric])
model.fit(tf.random.normal(shape=(100,10)), tf.random.normal(shape=(100,4)), verbose=0, epochs=10)

# saving checkpoint
saving_ckpt = tf.train.Checkpoint(model=model)
save_path = saving_ckpt.save('sequential')

# tracking(loading weights)
class Tracking(models.Model):
    def __init__(self):
        super(Tracking, self).__init__()
        self.layer = model.layers[1] # tracking target

# tracking checkpoint
tracking = Tracking()
tracking_ckpt = tf.train.Checkpoint(target=tracking)
save_path = tracking_ckpt.save('tracking')
tracking_ckpt_reader = tf.train.load_checkpoint(save_path)
tf.print(tf.train.list_variables('./')) #ckpt_reader.get_variable_to_dtype_map()

# define computational graph with model architecture
bias = tf.Variable(tf.zeros(shape=(4,)))
kernel = tf.Variable(tf.zeros(shape=(10, 4)))
layer = tf.train.Checkpoint(bias=bias, kernel=kernel)
target = tf.train.Checkpoint(layer=layer)
root = tf.train.Checkpoint(target=target)
root.restore(tf.train.latest_checkpoint('./'))
print(bias.numpy())
print(kernel.numpy())
```

##### Functional Model Training
`training flow`
```python
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics, callbacks, utils

inputs = layers.Input(shape=(None, 100))
x = layers.Dense(4, name='1L', activation="relu")(inputs)
x = layers.Dense(4, name='2L', activation="relu")(x)
outputs = layers.Dense(4, name='3L')(x)
model = models.Model(inputs, outputs)

# training instances
optimizer = optimizers.Adam(0.1)
mse_loss = losses.MeanSquaredError()
mae_metric = metrics.MeanAbsoluteError()
cb_lambda = callbacks.LambdaCallback(
    on_epoch_begin = lambda epoch, logs: print(f'[on_epoch_begin]: {epoch}, {logs}'),
    on_epoch_end   = lambda epoch, logs: print(f"[on_epoch_end]: {epoch}, {logs}"),
    on_batch_begin = lambda batch, logs: print(f"[on_batch_begin]: {batch}, {logs}"),
    on_batch_end   = lambda batch, logs: print(f"[on_batch_end]: {batch}, {logs}"),
    on_train_begin = lambda        logs: print(f"[on_train_begin]: {logs}"),
    on_train_end   = lambda        logs: print(f"[on_train_end]: {logs}"),)

# training
model.compile(optimizer=optimizer, loss=mse_loss, metrics=[mae_metric])
model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,4)), epochs=10, callbacks=[cb_lambda])
model.predict(tf.random.normal(shape=(1,100)))  # return prediction result
model.evaluate(tf.random.normal(shape=(1,100))) # return loss and metric 

tf.keras.backend.clear_session()
model.save("model/version/1")                            # creates a assets(folder), saved_model.pb(file), variables(folder) 
model.save_weights('model/version/1/variables/weights')  # creates a checkpoint, weights.data*, weights.index on the variables(folder)
model = models.load_model("model/version/1")
model.load_weights('model/version/1/variables/weights')  # check "tf.train.latest_checkpoint('model/version/1/variables')"
model.summary()

utils.plot_model(model, "model.png", show_shapes=True)
```

`training details`
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
model.layers
model.inputs
model.outputs

# model configuration& architecture saving
config = model.get_config()
model = models.Model.from_config(config)


# training
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.optimizer
model.loss

history = model.fit(tf.random.normal(shape=(100, 1, 1, 10)), tf.random.normal(shape=(100, 10)))
history.params
history.history
model.metrics

# prediction
model.predict(tf.random.normal(shape=(1, 1, 1, 10)))

# evaluation
model.evaluate(tf.random.normal(shape=(1, 1, 1, 10)))

# save & load (1: full model): Keras H5 format
model.save('model.h5')                      # creates a HDF5 file 'model.h5'
model = models.load_model('model.h5')
model.save_weights("weights.h5")
model.load_weights("weights.h5")
#model.add_loss(~)
#model.add_metric(~)

# model entities
model.submodules
model.submodules[-1].input
model.submodules[-1].output
model.variables
model.trainable_variables
model.non_trainable_variables

model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
```

#### Subclassing API Model
#### Subclassing Model Definition
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
history = model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,10)))
history.params
history.history

model.save("model/version/1/")
model.save_weights("model/version/1/")
model = models.load_model("model/version/1/")
model.load_weights("model/version/1/")
```

##### Subclassing Model Checkpoint
```python
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics

class Model(models.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.layer(x)

model = Model()
optimizer = optimizers.Adam(0.1)
mse_loss = losses.MeanSquaredError()
mae_metric = metrics.MeanAbsoluteError()
model.compile(optimizer=optimizer, loss=mse_loss, metrics=[mae_metric])
model.fit(tf.random.normal(shape=(100,10)), tf.random.normal(shape=(100,4)), verbose=0, epochs=10)

saving_ckpt = tf.train.Checkpoint(model=model)
save_path = saving_ckpt.save('subclassing')
tf.train.list_variables('./')
```
```
[('_CHECKPOINTABLE_OBJECT_GRAPH', []),
 ('model/layer/bias/.ATTRIBUTES/VARIABLE_VALUE', [4]),
 ('model/layer/bias/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE',
  [4]),
 ('model/layer/bias/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE',
  [4]),
 ('model/layer/kernel/.ATTRIBUTES/VARIABLE_VALUE', [10, 4]),
 ('model/layer/kernel/.OPTIMIZER_SLOT/model/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE',
  [10, 4]),
 ('model/layer/kernel/.OPTIMIZER_SLOT/model/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE',
  [10, 4]),
 ('model/optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('model/optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('model/optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('model/optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('model/optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE', []),
 ('save_counter/.ATTRIBUTES/VARIABLE_VALUE', [])]
```

`model tracking`
```python
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, losses, metrics

class Model(models.Model):
    def __init__(self):
        super(Model, self).__init__()
        self.layer = tf.keras.layers.Dense(4)

    def call(self, x):
        return self.layer(x)

model = Model()
optimizer = optimizers.Adam(0.1)
mse_loss = losses.MeanSquaredError()
mae_metric = metrics.MeanAbsoluteError()
model.compile(optimizer=optimizer, loss=mse_loss, metrics=[mae_metric])
model.fit(tf.random.normal(shape=(100,10)), tf.random.normal(shape=(100,4)), verbose=0, epochs=10)

saving_ckpt = tf.train.Checkpoint(model=model)
save_path = saving_ckpt.save('subclassing')
#tf.train.list_variables('./')

bias = tf.Variable(tf.zeros([4]))
print(bias.numpy())
layer = tf.train.Checkpoint(bias=bias)
model = tf.train.Checkpoint(layer=layer)
root = tf.train.Checkpoint(model=model)
status = root.restore(tf.train.latest_checkpoint('./'))
print(bias.numpy())
```

#### Subclassing Model Training
`training flow`
```python

```

`training details`
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
history = model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,10)))
history.params
history.history

# prediction
model.predict(tf.random.normal(shape=(1,100)))

# evaluation
model.evaluate(tf.random.normal(shape=(1,100)))

# model entities
model.layers
model.variables
model.trainable_variables
model.non_trainable_variables

# save & load(1): SavedModel Format
tf.keras.backend.clear_session()
model.save("model/version/1/")
model = models.load_model("model/version/1/")
model.save_weights("model/version/1/variables/weights") # creates a checkpoint, weights.data*, weights.index on the variables(folder)
model.load_weights("model/version/1/variables/weights") # check "tf.train.latest_checkpoint('model/version/1/variables')"


# model summary
model.summary()
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

# save & load(2): SavedModel Format
tf.keras.backend.clear_session()
tf.saved_model.save(model, "model/version/1/")
model = tf.saved_model.load("model/version/1/")
```


### train
- https://www.tensorflow.org/api_docs/python/tf/keras/callbacks
- https://www.tensorflow.org/api_docs/python/tf/train/load_checkpoint
- https://keras.io/guides/serialization_and_saving/

#### Checkpoint Objects
`Checkpoint API`
```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

model = models.Sequential()
model.add(layers.Dense(4, name='1L', activation="relu"))
optimizer = optimizers.Adam(0.1)

ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
ckpt.model
ckpt.optimizer
ckpt.restore(None)

ckpt.write('ckpts')
ckpt.read('ckpts')
ckpt.save('ckpts')
ckpt.restore('ckpts-1')

manager = tf.train.CheckpointManager(checkpoint=ckpt, directory='ckpts', max_to_keep=3)
manager.directory         # 'ckpts'
manager.checkpoint        # ckpt
manager.checkpoints       # max_to_keep
manager.latest_checkpoint # tf.train.latest_checkpoint(checkpoint_dir='ckpts')

reader = tf.train.load_checkpoint('ckpts')
variable_map = reader.get_variable_to_shape_map()
list_variables = list(map(lambda name: (name, variable_map[name]), variable_map.keys())) 

tf.train.list_variables(manager.latest_checkpoint)
tf.train.get_checkpoint_state(checkpoint_dir='ckpts')
```
`Checkpoint Usage`
```python
import tensorflow as tf

saving_ckpt = tf.train.Checkpoint()
saving_ckpt.variable_listwrap = [tf.Variable(2.)]
saving_ckpt.variable_dictwrap = {'target': saving_ckpt.variable_listwrap[0]}
save_path = saving_ckpt.save('variables')

restoring_ckpt = tf.train.Checkpoint()
new_variable = tf.Variable(0.)
assert 0. == new_variable.numpy()  # 아직 복구되지 않았습니다.
restoring_ckpt.variable_dictwrap = {'target': new_variable}
restoring_ckpt.restore(save_path)
assert 2. == new_variable.numpy()

tf.train.list_variables('./')
```

```python
import tensorflow as tf

saving_ckpt = tf.train.Checkpoint()
saving_ckpt.variable_listwrap = [tf.Variable(1.)]
saving_ckpt.variable_listwrap.append(tf.Variable(2.))
saving_ckpt.variable_dictwrap = {'origin': saving_ckpt.variable_listwrap[0]}
saving_ckpt.variable_dictwrap['synchronize'] = saving_ckpt.variable_listwrap[1]
save_path = saving_ckpt.save('variables')

restoring_ckpt = tf.train.Checkpoint()
new_variable = tf.Variable(0.)
assert 0. == new_variable.numpy()  # 아직 복구되지 않았습니다.
restoring_ckpt.variable_dictwrap = {'synchronize': new_variable}
restoring_ckpt.restore(save_path)
assert 2. == new_variable.numpy()

tf.train.list_variables('./')
```

#### Callback Checkpoint
`load_weights`
```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

model = models.Sequential()
model.add(layers.Dense(4, name='1L', activation="relu"))
optimizer = optimizers.Adam(0.1)
cp_model = callbacks.ModelCheckpoint(filepath='model/version/1/variables/variables', monitor='val_loss', save_best_only=False, save_weights_only=True, mode='auto', save_freq='epoch', options=None)

# training
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,4)), epochs=10, callbacks=[cp_model])
model.load_weights('model/version/1/variables/variables').assert_consumed()
```

`models.load_model`
```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

model = models.Sequential()
model.add(layers.Dense(4, name='1L', activation="relu"))
optimizer = optimizers.Adam(0.1)
cp_model = callbacks.ModelCheckpoint(filepath='model/version/1/', monitor='val_loss', save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', options=None)

# training
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,4)), epochs=10, callbacks=[cp_model])
model = models.load_model('model/version/1/')
```


#### Tensorflow Checkpoint
```python
import tensorflow as tf
from tensorflow.keras import optimizers

@tf.function
def Function(x):
    return W - x

W = tf.Variable([2.])
x = tf.constant([1.])
optimizer = optimizers.Adam(0.1)

ckpt = tf.train.Checkpoint(weight=W)
manager = tf.train.CheckpointManager(ckpt, 'model/version/1/variables', max_to_keep=3)

with tf.GradientTape() as tape:
    y = Function(x)    
gradient = tape.gradient(y, [W])
optimizer.apply_gradients(zip(gradient, [W]))
saving_path = manager.save()

tracing_variable = tf.Variable(tf.zeros(shape=(1,)))
tracing_ckpt = tf.train.Checkpoint(weight=tracing_variable)
tracing_ckpt.restore(manager.latest_checkpoint)

W.numpy() - tracing_variable.numpy()
```

```python
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers, metrics
from sklearn.model_selection import train_test_split

class Model(models.Model):
    def __init__(self, name=None):
        super(Model, self).__init__(name)
        self.layer1 = layers.Dense(10, activation='relu')
        self.layer2 = layers.Dense(10, activation='relu')
        self.layer3 = layers.Dense(2, activation='linear')

    def call(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# [Dataset]
X = tf.random.normal(shape=(1000, 10))
y = tf.random.normal(shape=(1000, 2))
train_X, test_X, train_y, test_y = train_test_split(X.numpy(), y.numpy(), test_size=0.3, shuffle=False)
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y)).repeat(10).batch(10)
test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_y)).batch(1)

# [Modeling Instances]
model = Model()
criterion = losses.MeanSquaredError()
optimizer = optimizers.Adam(0.1)
train_loss = metrics.Mean(name='train_loss')
test_loss = metrics.Mean(name='test_loss')

# [Saving Instantces]
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(ckpt, 'model/version/1/variables', max_to_keep=3)
ckpt.restore(manager.latest_checkpoint)

# [Train Step]
@tf.function
def train_step(features, targets):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        cost = criterion(targets, predictions)
    gradients = tape.gradient(cost, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(cost)
    
# [Test Step]
@tf.function
def test_step(features, targets):
    predictions = model(features, training=False)
    cost = criterion(targets, predictions)
    test_loss(cost)
    
EPOCHS = 5
saving_paths = list()
for epoch in range(EPOCHS):
    train_loss.reset_states()
    test_loss.reset_states()

    for features, targets in train_dataset:
        train_step(features, targets)
        ckpt.step.assign_add(1)
        if ckpt.step % 100 == 0:
            saving_path = manager.save() # return saving_path
            saving_paths.append((ckpt.step, saving_path))
            
    for features, targets in test_dataset:
        test_step(features, targets)

    print(
        f'[Epoch {epoch + 1}][CKPT {ckpt.step.numpy()}], '
        f'Train Loss: {train_loss.result()}, '
        f'Test Loss: {test_loss.result()}, '
    )

# [Model Summary]
model.summary()

# [Model Save&Load]
tf.saved_model.save(model, "model/version/1/")
model = tf.saved_model.load("model/version/1/")
```

### Callbacks
- https://www.tensorflow.org/guide/keras/custom_callback
- https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/

```python
import tensorflow as tf
from tensorflow.keras import callbacks

cb_csvlogger = callbacks.CSVLogger(filename='log.csv')
cb_earlystopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
cb_lambda = callbacks.LambdaCallback(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None)
cb_scheduler = callbacks.LearningRateScheduler(schedule=lambda epoch, lr: lr if epoch < 10 else lr * tf.math.exp(-0.1), verbose=0)
cb_checkpoint = callbacks.ModelCheckpoint('ckpt', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', options=None, initial_value_threshold=None)
cb_progbar = callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)
cb_rlrp = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
cb_remote = callbacks.RemoteMonitor(root='http://localhost:9000', path='/publish/epoch/end/', field='data', headers=None, send_as_json=False)
cb_tensorboard = callbacks.TensorBoard(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False, write_steps_per_second=False, update_freq='epoch', profile_batch=0, embeddings_freq=0, embeddings_metadata=None)
cb_terminate = callbacks.TerminateOnNaN()
```

#### EarlyStopping
```python
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks

model = models.Sequential()
model.add(layers.Dense(4, name='1L', activation="relu"))
model.add(layers.Dense(4, name='2L', activation="relu"))
model.add(layers.Dense(4, name='3L'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

callback = callbacks.EarlyStopping(monitor='loss', patience=3)
history = model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,4)), epochs=10, callbacks=[callback])
```

#### LambdaCallback
```python
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks

model = models.Sequential()
model.add(layers.Dense(4, name='1L', activation="relu"))
model.add(layers.Dense(4, name='2L', activation="relu"))
model.add(layers.Dense(4, name='3L'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

callback = callbacks.LambdaCallback(
    on_epoch_begin = lambda epoch, logs: print(f'[on_epoch_begin]: {epoch}, {logs}'),
    on_epoch_end   = lambda epoch, logs: print(f"[on_epoch_end]: {epoch}, {logs}"),
    on_batch_begin = lambda batch, logs: print(f"[on_batch_begin]: {batch}, {logs}"),
    on_batch_end   = lambda batch, logs: print(f"[on_batch_end]: {batch}, {logs}"),
    on_train_begin = lambda        logs: print(f"[on_train_begin]: {logs}"),
    on_train_end   = lambda        logs: print(f"[on_train_end]: {logs}"),)
history = model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,4)), epochs=1, verbose=0,  callbacks=[callback])
```
```
[on_train_begin]: {}
[on_epoch_begin]: 0, {}
[on_batch_begin]: 0, {}
[on_batch_end  ]: 0, {'loss': 1.7096302509307861, 'mae': 1.0549283027648926}
[on_batch_begin]: 1, {}
[on_batch_end  ]: 1, {'loss': 1.7188581228256226, 'mae': 1.0462791919708252}
[on_batch_begin]: 2, {}
[on_batch_end  ]: 2, {'loss': 1.7125993967056274, 'mae': 1.0388036966323853}
[on_batch_begin]: 3, {}
[on_batch_end  ]: 3, {'loss': 1.6923072338104248, 'mae': 1.0348076820373535}
[on_epoch_end  ]: 0, {'loss': 1.6923072338104248, 'mae': 1.0348076820373535}
[on_train_end  ]: {'loss': 1.6923072338104248, 'mae': 1.0348076820373535}
```

#### LearningRateScheduler
```python
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks

model = models.Sequential()
model.add(layers.Dense(4, name='1L', activation="relu"))
model.add(layers.Dense(4, name='2L', activation="relu"))
model.add(layers.Dense(4, name='3L'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

callback = callbacks.LearningRateScheduler(schedule=lambda epoch, lr: lr if epoch < 10 else lr * tf.math.exp(-0.1), verbose=0)
history = model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,4)), epochs=10, callbacks=[callback])
```

#### ModelCheckpoint
```python
import tensorflow as tf
from tensorflow.keras import models, layers, callbacks

model = models.Sequential()
model.add(layers.Dense(4, name='1L', activation="relu"))
model.add(layers.Dense(4, name='2L', activation="relu"))
model.add(layers.Dense(4, name='3L'))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

callback = callbacks.ModelCheckpoint('ckpt', monitor='loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch', options=None, initial_value_threshold=None)
history = model.fit(tf.random.normal(shape=(100,100)), tf.random.normal(shape=(100,4)), epochs=10, callbacks=[callback])
```


#### Custom Callback
```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

class CustomCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print(f"[{epoch}][on_epoch_begin][Start epoch of training]: {keys}")

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print(f"[{epoch}][on_epoch_end][End epoch of training]: {keys}")
        print(f"    > {logs['loss']}, {logs['mae']}, {logs['val_loss']}, {logs['val_mae']}")
    
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_train_begin][Starting training]: {keys}")

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_train_end][Stop training]: {keys}")
        print(f"    > {logs['loss']}, {logs['mae']}, {logs['val_loss']}, {logs['val_mae']}")

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_train_batch_begin][...Training: start of batch]: {keys}")

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_train_batch_end][...Training: start of batch]: {keys}")
        print(f"    > {logs['loss']}, {logs['mae']}")        
        
    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_test_begin][Start testing]: {keys}")

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_test_end][Stop testing]: {keys}")
        print(f"    > {logs['loss']}, {logs['mae']}")
                
    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_test_batch_begin][...Evaluating: start of batch]: {keys}")

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_test_batch_end][...Evaluating: end of batch]: {keys}")
        print(f"    > {logs['loss']}, {logs['mae']}")        

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_predict_begin][Start predicting]: {keys}")

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_predict_end][Stop predicting]: {keys}")

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_predict_batch_begin][...Predicting: start of batch]: {keys}")

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_predict_batch_end][...Predicting: end of batch]: {keys}")
        print(f"    > {logs['outputs'][0]}")

# training
model = models.Sequential(layers.Dense(4, name='1L', activation="relu"))
model.compile(optimizer='adam', loss="mse", metrics=["mae"])
history = model.fit(
    x=tf.random.normal(shape=(100,100)), y=tf.random.normal(shape=(100,4)), 
    batch_size=10, epochs=1, verbose=0,
    validation_split=0.2, validation_batch_size=10, 
    callbacks=[CustomCallback()])
```
```
   [on_train_begin      ][Starting training            ]: []
[0][on_epoch_begin      ][Start epoch of training      ]: []
[0][on_train_batch_begin][...Training: start of batch  ]: []
[0][on_train_batch_end  ][...Training: start of batch  ]: ['loss', 'mae']
[1][on_train_batch_begin][...Training: start of batch  ]: []
[1][on_train_batch_end  ][...Training: start of batch  ]: ['loss', 'mae']
[2][on_train_batch_begin][...Training: start of batch  ]: []
[2][on_train_batch_end  ][...Training: start of batch  ]: ['loss', 'mae']
[3][on_train_batch_begin][...Training: start of batch  ]: []
[3][on_train_batch_end  ][...Training: start of batch  ]: ['loss', 'mae']
[4][on_train_batch_begin][...Training: start of batch  ]: []
[4][on_train_batch_end  ][...Training: start of batch  ]: ['loss', 'mae']
[5][on_train_batch_begin][...Training: start of batch  ]: []
[5][on_train_batch_end  ][...Training: start of batch  ]: ['loss', 'mae']
[6][on_train_batch_begin][...Training: start of batch  ]: []
[6][on_train_batch_end  ][...Training: start of batch  ]: ['loss', 'mae']
[7][on_train_batch_begin][...Training: start of batch  ]: []
[7][on_train_batch_end  ][...Training: start of batch  ]: ['loss', 'mae']
   [on_test_begin       ][Start testing                ]: []
[0][on_test_batch_begin ][...Evaluating: start of batch]: []
[0][on_test_batch_end   ][...Evaluating: end of batch  ]: ['loss', 'mae']
[1][on_test_batch_begin ][...Evaluating: start of batch]: []
[1][on_test_batch_end   ][...Evaluating: end of batch  ]: ['loss', 'mae']
   [on_test_end         ][Stop testing                 ]: ['loss', 'mae']
[0][on_epoch_end        ][End epoch of training        ]: ['loss', 'mae', 'val_loss', 'val_mae']
   [on_train_end        ][Stop training                ]: ['loss', 'mae', 'val_loss', 'val_mae']
```

`learning callback`
```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

class CustomCallback(callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print(f"[{epoch}][on_epoch_begin][Start epoch of training]: {keys}")

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print(f"[{epoch}][on_epoch_end][End epoch of training]: {keys}")
        print(f"    > {logs['loss']}, {logs['mae']}, {logs['val_loss']}, {logs['val_mae']}")
    
# training
model = models.Sequential(layers.Dense(4, name='1L', activation="relu"))
model.compile(optimizer='adam', loss="mse", metrics=["mae"])
history = model.fit(
    x=tf.random.normal(shape=(100,100)), y=tf.random.normal(shape=(100,4)), 
    batch_size=10, epochs=1, verbose=0,
    validation_split=0.2, validation_batch_size=10, 
    callbacks=[CustomCallback()])
```
```
[0][on_epoch_begin][Start epoch of training]: []
[0][on_epoch_end  ][End epoch of training  ]: ['loss', 'mae', 'val_loss', 'val_mae']
    > 1.8481264114379883, 1.0672204494476318, 1.496233344078064, 0.9514732360839844
```

`training callback`
```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

class CustomCallback(callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_train_begin][Starting training]: {keys}")

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_train_end][Stop training]: {keys}")
        print(f"    > {logs['loss']}, {logs['mae']}, {logs['val_loss']}, {logs['val_mae']}")

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_train_batch_begin][...Training: start of batch]: {keys}")

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_train_batch_end][...Training: start of batch]: {keys}")
        print(f"    > {logs['loss']}, {logs['mae']}")

# training
model = models.Sequential(layers.Dense(4, name='1L', activation="relu"))
model.compile(optimizer='adam', loss="mse", metrics=["mae"])
history = model.fit(
    x=tf.random.normal(shape=(100,100)), y=tf.random.normal(shape=(100,4)), 
    batch_size=10, epochs=1, verbose=0,
    validation_split=0.2, validation_batch_size=10, 
    callbacks=[CustomCallback()])
```
```
   [on_train_begin      ][Starting training          ]: []
[0][on_train_batch_begin][...Training: start of batch]: []
[0][on_train_batch_end  ][...Training: start of batch]: ['loss', 'mae']
    > 1.9303505420684814, 1.1057828664779663
[1][on_train_batch_begin][...Training: start of batch]: []
[1][on_train_batch_end  ][...Training: start of batch]: ['loss', 'mae']
    > 2.3909990787506104, 1.1519211530685425
[2][on_train_batch_begin][...Training: start of batch]: []
[2][on_train_batch_end  ][...Training: start of batch]: ['loss', 'mae']
    > 2.1300878524780273, 1.093126893043518
[3][on_train_batch_begin][...Training: start of batch]: []
[3][on_train_batch_end  ][...Training: start of batch]: ['loss', 'mae']
    > 1.9822704792022705, 1.0604379177093506
[4][on_train_batch_begin][...Training: start of batch]: []
[4][on_train_batch_end  ][...Training: start of batch]: ['loss', 'mae']
    > 1.8501325845718384, 1.040222406387329
[5][on_train_batch_begin][...Training: start of batch]: []
[5][on_train_batch_end  ][...Training: start of batch]: ['loss', 'mae']
    > 1.9133890867233276, 1.0388209819793701
[6][on_train_batch_begin][...Training: start of batch]: []
[6][on_train_batch_end  ][...Training: start of batch]: ['loss', 'mae']
    > 2.0827043056488037, 1.0907970666885376
[7][on_train_batch_begin][...Training: start of batch]: []
[7][on_train_batch_end  ][...Training: start of batch]: ['loss', 'mae']
    > 1.9632031917572021, 1.069227695465088
   [on_train_end        ][Stop training              ]: ['loss', 'mae', 'val_loss', 'val_mae']
    > 1.9632031917572021, 1.069227695465088, 1.8367271423339844, 1.0263046026229858
```

`validation callback`
```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

class CustomCallback(callbacks.Callback):
    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_test_begin][Start testing]: {keys}")

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_test_end][Stop testing]: {keys}")
        print(f"    > {logs['loss']}, {logs['mae']}")
                
    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_test_batch_begin][...Evaluating: start of batch]: {keys}")

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_test_batch_end][...Evaluating: end of batch]: {keys}")
        print(f"    > {logs['loss']}, {logs['mae']}")        

# training
model = models.Sequential(layers.Dense(4, name='1L', activation="relu"))
model.compile(optimizer='adam', loss="mse", metrics=["mae"])
history = model.fit(
    x=tf.random.normal(shape=(100,100)), y=tf.random.normal(shape=(100,4)), 
    batch_size=10, epochs=1, verbose=0,
    validation_split=0.2, validation_batch_size=10, 
    callbacks=[CustomCallback()])
```
```
   [on_test_begin      ][Start testing                ]: []
[0][on_test_batch_begin][...Evaluating: start of batch]: []
[0][on_test_batch_end  ][...Evaluating: end of batch  ]: ['loss', 'mae']
    > 1.9301878213882446, 1.2562932968139648
[1][on_test_batch_begin][...Evaluating: start of batch]: []
[1][on_test_batch_end  ][...Evaluating: end of batch  ]: ['loss', 'mae']
    > 2.341826915740967, 1.2110979557037354
   [on_test_end        ][Stop testing                 ]: ['loss', 'mae']
    > 2.341826915740967, 1.2110979557037354
```

`prediction callback`
```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

class CustomCallback(callbacks.Callback):        
    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_predict_begin][Start predicting]: {keys}")

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_predict_end][Stop predicting]: {keys}")

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_predict_batch_begin][...Predicting: start of batch]: {keys}")

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_predict_batch_end][...Predicting: end of batch]: {keys}")
        print(f"    > {logs['outputs'][0]}")

# training
model = models.Sequential(layers.Dense(4, name='1L', activation="relu"))
model.compile(optimizer='adam', loss="mse", metrics=["mae"])
history = model.fit(
    x=tf.random.normal(shape=(100,100)), y=tf.random.normal(shape=(100,4)), 
    batch_size=10, epochs=1, verbose=0,
    validation_split=0.2, validation_batch_size=10, 
    callbacks=[CustomCallback()])

model.predict(tf.random.normal(shape=(100,100)), batch_size=10, callbacks=[CustomCallback()])
```
```
   [on_predict_begin      ][Start predicting             ]: []
[0][on_predict_batch_begin][...Predicting: start of batch]: []
[0][on_predict_batch_end  ][...Predicting: end of batch  ]: ['outputs']
    > [0.5502743 1.4684863 0.        1.505536 ]
[1][on_predict_batch_begin][...Predicting: start of batch]: []
[1][on_predict_batch_end  ][...Predicting: end of batch  ]: ['outputs']
    > [2.1667542 2.1442165 0.        0.       ]
[2][on_predict_batch_begin][...Predicting: start of batch]: []
[2][on_predict_batch_end  ][...Predicting: end of batch  ]: ['outputs']
    > [0.        2.0003762 2.1422117 1.6096756]
[3][on_predict_batch_begin][...Predicting: start of batch]: []
[3][on_predict_batch_end  ][...Predicting: end of batch  ]: ['outputs']
    > [0.6158651 0.        0.        0.       ]
[4][on_predict_batch_begin][...Predicting: start of batch]: []
[4][on_predict_batch_end  ][...Predicting: end of batch  ]: ['outputs']
    > [0.        0.8896242 0.6119127 0.       ]
[5][on_predict_batch_begin][...Predicting: start of batch]: []
[5][on_predict_batch_end  ][...Predicting: end of batch  ]: ['outputs']
    > [1.7670332  0.62246895 0.         0.        ]
[6][on_predict_batch_begin][...Predicting: start of batch]: []
[6][on_predict_batch_end  ][...Predicting: end of batch  ]: ['outputs']
    > [0. 0. 0. 0.]
[7][on_predict_batch_begin][...Predicting: start of batch]: []
[7][on_predict_batch_end  ][...Predicting: end of batch  ]: ['outputs']
    > [1.71011   1.0592731 0.        1.487398 ]
[8][on_predict_batch_begin][...Predicting: start of batch]: []
[8][on_predict_batch_end  ][...Predicting: end of batch  ]: ['outputs']
    > [2.613045   1.1312315  1.1647196  0.24384966]
[9][on_predict_batch_begin][...Predicting: start of batch]: []
[9][on_predict_batch_end  ][...Predicting: end of batch  ]: ['outputs']
    > [0.652605   0.         0.         0.17018515]
   [on_predict_end        ][Stop predicting              ]: []
```

`evaluation callback`
```python
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

class CustomCallback(callbacks.Callback):
    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_test_begin][Start testing]: {keys}")

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        print(f"[on_test_end][Stop testing]: {keys}")
        print(f"    > {logs['loss']}, {logs['mae']}")
                
    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_test_batch_begin][...Evaluating: start of batch]: {keys}")

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        print(f"[{batch}][on_test_batch_end][...Evaluating: end of batch]: {keys}")
        print(f"    > {logs['loss']}, {logs['mae']}")        

# training
model = models.Sequential(layers.Dense(4, name='1L', activation="relu"))
model.compile(optimizer='adam', loss="mse", metrics=["mae"])
history = model.fit(
    x=tf.random.normal(shape=(100,100)), y=tf.random.normal(shape=(100,4)), 
    batch_size=10, epochs=1, verbose=0,
    validation_split=0.2, validation_batch_size=10)
model.evaluate(x=tf.random.normal(shape=(100,100)), y=tf.random.normal(shape=(100,4)), batch_size=10, verbose=0, callbacks=[CustomCallback()])    
```
```
   [on_test_begin      ][Start testing                ]: []
[0][on_test_batch_begin][...Evaluating: start of batch]: []
[0][on_test_batch_end  ][...Evaluating: end of batch  ]: ['loss', 'mae']
    > 1.417934775352478, 0.9490009546279907
[1][on_test_batch_begin][...Evaluating: start of batch]: []
[1][on_test_batch_end  ][...Evaluating: end of batch  ]: ['loss', 'mae']
    > 1.7772287130355835, 1.0406653881072998
[2][on_test_batch_begin][...Evaluating: start of batch]: []
[2][on_test_batch_end  ][...Evaluating: end of batch  ]: ['loss', 'mae']
    > 2.074294090270996, 1.143292784690857
[3][on_test_batch_begin][...Evaluating: start of batch]: []
[3][on_test_batch_end  ][...Evaluating: end of batch  ]: ['loss', 'mae']
    > 2.0633292198181152, 1.1475474834442139
[4][on_test_batch_begin][...Evaluating: start of batch]: []
[4][on_test_batch_end  ][...Evaluating: end of batch  ]: ['loss', 'mae']
    > 2.107672929763794, 1.1466442346572876
[5][on_test_batch_begin][...Evaluating: start of batch]: []
[5][on_test_batch_end  ][...Evaluating: end of batch  ]: ['loss', 'mae']
    > 2.0635533332824707, 1.1400113105773926
[6][on_test_batch_begin][...Evaluating: start of batch]: []
[6][on_test_batch_end  ][...Evaluating: end of batch  ]: ['loss', 'mae']
    > 2.014301300048828, 1.1164398193359375
[7][on_test_batch_begin][...Evaluating: start of batch]: []
[7][on_test_batch_end  ][...Evaluating: end of batch  ]: ['loss', 'mae']
    > 2.0907540321350098, 1.142563819885254
[8][on_test_batch_begin][...Evaluating: start of batch]: []
[8][on_test_batch_end  ][...Evaluating: end of batch  ]: ['loss', 'mae']
    > 2.041452646255493, 1.1321040391921997
[9][on_test_batch_begin][...Evaluating: start of batch]: []
[9][on_test_batch_end  ][...Evaluating: end of batch  ]: ['loss', 'mae']
    > 2.0367326736450195, 1.1176493167877197
   [on_test_end        ][Stop testing                 ]: ['loss', 'mae']
    > 2.0367326736450195, 1.1176493167877197
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
iterable_dataset.element_spec
iterable_dataset.enumerate()

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

iterable_dataset = tf.data.Dataset.range(1, 11)  # ==> [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
iterable_dataset = iterable_dataset.interleave(lambda x: tf.data.Dataset.from_tensors(x).repeat(20), cycle_length=4, block_length=7)
iterable_dataset.element_spec
iterable_dataset.enumerate()

display(pd.DataFrame({ k:[v] for k,v in Counter(iterable_dataset.as_numpy_iterator()).items()}).T.rename(columns={0:'CNT'}))
print('TOTALNUM:', len(list(iterable_dataset.as_numpy_iterator())))
print('ELEMENTS:', list(iterable_dataset.as_numpy_iterator()))
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
iterable_dataset.element_spec
iterable_dataset.enumerate()

list(iterable_dataset.as_numpy_iterator())
```
```python
import tensorflow as tf

def preprocessing(x):
    return x+1

iterable_dataset = tf.data.Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
iterable_dataset = iterable_dataset.map(preprocessing)
iterable_dataset.element_spec
iterable_dataset.enumerate()

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
iterable_dataset.element_spec
iterable_dataset.enumerate()

dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

size = 2
iterable_dataset = tf.data.Dataset.range(size).repeat(count=2)
iterable_dataset.element_spec
iterable_dataset.enumerate()

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
iterable_dataset.element_spec
iterable_dataset.enumerate()

dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

size = 5
iterable_dataset = tf.data.Dataset.range(size).batch(batch_size=2, drop_remainder=False)
iterable_dataset.element_spec
iterable_dataset.enumerate()

dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

size = 6
iterable_dataset = tf.data.Dataset.range(size).shuffle(buffer_size=size).batch(batch_size=2)
iterable_dataset.element_spec
iterable_dataset.enumerate()

dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

size = 6
iterable_dataset = tf.data.Dataset.range(size).shuffle(buffer_size=size).batch(batch_size=2)
iterable_dataset.element_spec
iterable_dataset.enumerate()

dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

size = 6
iterable_dataset = tf.data.Dataset.range(size).repeat(count=3).shuffle(buffer_size=size).batch(batch_size=2)
iterable_dataset.element_spec
iterable_dataset.enumerate()

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
iterable_dataset.element_spec
iterable_dataset.enumerate()

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
iterable_dataset.element_spec
iterable_dataset.enumerate()

for batch in iterable_dataset.batch(4).take(4):
    print([arr.numpy() for arr in batch])
```


`tf.data.Dataset.from_tensors`
```python
import tensorflow as tf

iterable_dataset = tf.data.Dataset.from_tensors(tf.constant(100))
iterable_dataset.element_spec
iterable_dataset.enumerate()

dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
```
```python
import tensorflow as tf

iterable_dataset = tf.data.Dataset.from_tensors(tf.constant(100)).repeat(2)
iterable_dataset.element_spec
iterable_dataset.enumerate()

dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
```

`tf.data.Dataset.from_tensor_slices`
```python
import tensorflow as tf

dataset_ = tf.random.normal(shape=(3,5))
iterable_dataset = tf.data.Dataset.from_tensor_slices((dataset_, ))
iterable_dataset.element_spec
iterable_dataset.enumerate()

list(iterable_dataset.as_numpy_iterator())
list(iterable_dataset)
```
```python
import tensorflow as tf

dataset_ = tf.random.normal(shape=(3,5))
iterable_dataset = tf.data.Dataset.from_tensor_slices((dataset_, ))
iterable_dataset.element_spec
iterable_dataset.enumerate()

dataset_iterator = iter(iterable_dataset)
dataset_iterator.get_next()
dataset_iterator.get_next()
dataset_iterator.get_next()
```
```python
import tensorflow as tf

iterable_dataset = tf.data.Dataset.from_tensor_slices(['a', 'b', 'c'])
iterable_dataset.element_spec
iterable_dataset.enumerate()

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
iterable_dataset.element_spec
iterable_dataset.enumerate()

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
iterable_dataset.element_spec
iterable_dataset.enumerate()

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
iterable_dataset.element_spec
iterable_dataset.enumerate()

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
iterable_dataset.element_spec
iterable_dataset.enumerate()

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
    # OUTPUT: (indices, features)    
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
    # OUTPUT: (indices, features)    
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
    # OUTPUT: (indices, features)    
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



# [Train Step]
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
    
# [Test Step]
@tf.function
def test_step(features, targets):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(features, training=False)
    cost = criterion(targets, predictions)

    test_loss(cost)
    test_accuracy(targets, predictions)

# [Training]
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

`Layer API`
```python
import tensorflow as tf
from tensorflow.keras import layers

layer = layers.AbstractRNNCell()
layer = layers.Activation( activation='linear' )
layer = layers.ActivityRegularization()
layer = layers.Add()
layer = layers.AdditiveAttention()
layer = layers.AlphaDropout( rate=0.1 )
layer = layers.Attention()
layer = layers.Average()
layer = layers.AveragePooling1D()
layer = layers.AveragePooling2D()
layer = layers.AveragePooling3D()
layer = layers.AvgPool1D()
layer = layers.AvgPool2D()
layer = layers.AvgPool3D()
layer = layers.BatchNormalization()
layer = layers.Bidirectional( layer=layers.SimpleRNN(units=32) )
layer = layers.CategoryEncoding( num_tokens=32 )
layer = layers.CenterCrop( height=80, width=60 )
layer = layers.Concatenate()
layer = layers.Conv1D( filters=32, kernel_size=3 )
layer = layers.Conv1DTranspose( filters=32, kernel_size=3 )
layer = layers.Conv2D( filters=2, kernel_size=3 )
layer = layers.Conv2DTranspose(  filters=2, kernel_size=3 )
layer = layers.Conv3D( filters=2, kernel_size=3 )
layer = layers.Conv3DTranspose( filters=2, kernel_size=3 )
layer = layers.ConvLSTM1D( filters=32, kernel_size=3 )
layer = layers.ConvLSTM2D( filters=2, kernel_size=3 )
layer = layers.ConvLSTM3D( filters=2, kernel_size=3 )
layer = layers.Cropping1D()
layer = layers.Cropping2D()
layer = layers.Cropping3D()
layer = layers.Dense( units=32 )
layer = layers.DenseFeatures( feature_columns=[tf.feature_column.numeric_column('COL1'), tf.feature_column.numeric_column('COL2'), ] )
layer = layers.DepthwiseConv1D( kernel_size=3 )
layer = layers.DepthwiseConv2D( kernel_size=3 )
layer = layers.Discretization()
layer = layers.Dot( axes=1 )
layer = layers.Dropout( rate=0.1 )
layer = layers.ELU()
layer = layers.Embedding( input_dim=8, output_dim=32 )
layer = layers.Flatten()
layer = layers.GRU( units=32 )
layer = layers.GRUCell( units=32 )
layer = layers.GaussianDropout( rate=0.1 )
layer = layers.GaussianNoise( stddev=1 )
layer = layers.GlobalAveragePooling1D()
layer = layers.GlobalAveragePooling2D()
layer = layers.GlobalAveragePooling3D()
layer = layers.GlobalAvgPool1D()
layer = layers.GlobalAvgPool2D()
layer = layers.GlobalAvgPool3D()
layer = layers.GlobalMaxPool1D()
layer = layers.GlobalMaxPool2D()
layer = layers.GlobalMaxPool3D()
layer = layers.GlobalMaxPooling1D()
layer = layers.GlobalMaxPooling2D()
layer = layers.GlobalMaxPooling3D()
layer = layers.GlobalMaxPooling2D()
layer = layers.GlobalMaxPooling3D()
layer = layers.Hashing( num_bins=10 )
layer = layers.InputLayer()
layer = layers.InputSpec()
layer = layers.IntegerLookup()
layer = layers.LSTM( units=32 )
layer = layers.LSTMCell( units=32 )
layer = layers.Lambda( function=lambda x: x+1 )
layer = layers.Layer()
layer = layers.LayerNormalization()
layer = layers.LeakyReLU()
layer = layers.LocallyConnected1D( filters=2, kernel_size=3 )
layer = layers.LocallyConnected2D( filters=2, kernel_size=3 )
layer = layers.Masking()
layer = layers.MaxPool1D()
layer = layers.MaxPool2D()
layer = layers.MaxPool3D()
layer = layers.MaxPooling1D()
layer = layers.MaxPooling2D()
layer = layers.MaxPooling3D()
layer = layers.Maximum()
layer = layers.Minimum()
layer = layers.MultiHeadAttention( num_heads=8, key_dim=512 )
layer = layers.Multiply()
layer = layers.Normalization()
layer = layers.PReLU()
layer = layers.Permute( dims=(2, 1) )
layer = layers.RNN( cell=layers.SimpleRNNCell(units=32) )
layer = layers.RandomContrast( factor=.2 )
layer = layers.RandomCrop( height=80, width=60 )
layer = layers.RandomFlip()
layer = layers.RandomHeight( factor=.2 )
layer = layers.RandomRotation( factor=.2 )
layer = layers.RandomTranslation( height_factor=.2, width_factor=.2 )
layer = layers.RandomWidth( factor=.2 )
layer = layers.RandomZoom( height_factor=.2 )
layer = layers.ReLU()
layer = layers.RepeatVector( n=3 )
layer = layers.Rescaling( scale=3 )
layer = layers.Reshape( target_shape=(10,10) )
layer = layers.Resizing( height=80, width=60 )
layer = layers.SeparableConv1D( filters=2, kernel_size=3 )
layer = layers.SeparableConv2D( filters=2, kernel_size=3 )
layer = layers.SimpleRNN( units=32 )
layer = layers.SimpleRNNCell( units=32 )
layer = layers.Softmax()
layer = layers.SpatialDropout1D( rate=.1 )
layer = layers.SpatialDropout2D( rate=.1 )
layer = layers.SpatialDropout3D( rate=.1 )
layer = layers.StackedRNNCells( cells=[layers.SimpleRNNCell(units=32) for _ in range(2)])
layer = layers.StringLookup()
layer = layers.Subtract()
layer = layers.TextVectorization()
layer = layers.ThresholdedReLU()
layer = layers.TimeDistributed( layer=layers.Dense(units=32) )
layer = layers.UpSampling1D()
layer = layers.UpSampling2D()
layer = layers.UpSampling3D()
#layer = layers.Wrapper()
layer = layers.ZeroPadding1D()
layer = layers.ZeroPadding2D()
layer = layers.ZeroPadding3D()
```

`Layer Architecture Save&Load: only architecture except for trainable parameters`
```python
from tensorflow.keras import layers

layer = layers.Dense( units=32 )         # define
config = layer.get_config()              # save
layer = layers.Dense.from_config(config) # load
```

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
- https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer  

#### Built-in Optimizer Definition
```python
import json
from tensorflow.keras import optimizers  

optimizer = optimizers.Adamax(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax')

# save&load in memory
config = optimizers.serialize(optimizer)
optimizer = optimizers.deserialize(config)
optimizer = optimizer.from_config(config['config'])
```
```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers  

optimizer = optimizers.Adamax(learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Adamax')
model = models.Sequential(layers.Dense(units=32))
model.compile(optimizer=optimizer, loss='mse')
model.fit(tf.random.normal(shape=(1000,64)), tf.random.normal(shape=(1000,32)))

config_details = optimizer.get_weights()
optimizer.set_weights(config_details)
```

#### Optimizer Schduler
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
losses.binary_crossentropy(target, hypothesis, from_logits=False)
tf.math.reduce_mean(-tf.math.log([(1-0.3), 0.8, 0.5, 0.2]))                             # from_logits=False
tf.math.reduce_mean(-tf.math.log([(1-0.57444252), 0.68997448, 0.62245933, 0.549834  ])) # from_logits=True, 1/(1 + np.exp(-1*np.array([0.3, 0.8, 0.5, 0.2])))

target = [[0, 1, 0], [0, 0, 1]]
hypothesis = [[0.05, 0.95, 0], [0.2, 0.7, 0.1]]
cost = losses.CategoricalCrossentropy(from_logits=False)
cost(target, hypothesis)
losses.categorical_crossentropy(target, hypothesis, from_logits=False).numpy().mean()
tf.math.reduce_mean(-tf.math.log([0.95, 0.1]))                                                            # from_logits=False
tf.math.reduce_mean(-tf.math.log([tf.nn.softmax([0.05, 0.95, 0])[1], tf.nn.softmax([0.2, 0.7, 0.1])[2]])) # from_logits=True

target = [1, 2]
hypothesis = [[0.05, 0.95, 0], [0.2, 0.7, 0.1]]
cost = losses.SparseCategoricalCrossentropy(from_logits=True)
cost(target, hypothesis)
losses.sparse_categorical_crossentropy(target, hypothesis, from_logits=True).numpy().mean()
tf.math.reduce_mean(-tf.math.log([0.95, 0.1]))                                                            # from_logits=False
tf.math.reduce_mean(-tf.math.log([tf.nn.softmax([0.05, 0.95, 0])[1], tf.nn.softmax([0.2, 0.7, 0.1])[2]])) # from_logits=True

target = [[0, 1], [0, 0]]
hypothesis = [[0.6, 0.4], [0.4, 0.6]]
cost = losses.CategoricalHinge()
cost(target, hypothesis) 
losses.categorical_hinge(target, hypothesis).numpy().mean()

target = [[0., 1.], [1., 1.]]
hypothesis = [[1., 0.], [1., 1.]]
cost = losses.CosineSimilarity(axis=-1)
cost(target, hypothesis) 
losses.cosine_similarity(target, hypothesis).numpy().mean()

target = [[0., 1.], [0., 0.]]
hypothesis = [[0.6, 0.4], [0.4, 0.6]]
cost = losses.Hinge()
cost(target, hypothesis) 
losses.hinge(target, hypothesis).numpy().mean()

target = [[0, 1], [0, 0]]
hypothesis = [[0.6, 0.4], [0.4, 0.6]]
cost = losses.Huber()
cost(target, hypothesis) 
losses.huber(target, hypothesis).numpy().mean()

target = [[0, 1], [0, 0]]
hypothesis = [[0.6, 0.4], [0.4, 0.6]]
cost = losses.KLDivergence()
cost(target, hypothesis) 
losses.kl_divergence(target, hypothesis).numpy().mean()

target = [[0., 1.], [0., 0.]]
hypothesis = [[1., 1.], [0., 0.]]
cost = losses.LogCosh()
cost(target, hypothesis) 
losses.logcosh(target, hypothesis).numpy().mean()

target = [[0., 1.], [0., 0.]]
hypothesis = [[1., 1.], [1., 0.]]
cost = losses.MeanAbsoluteError()
cost(target, hypothesis) 
losses.mean_absolute_error(target, hypothesis).numpy().mean()

target = [[2., 1.], [2., 3.]]
hypothesis = [[1., 1.], [1., 0.]]
cost = losses.MeanAbsolutePercentageError()
cost(target, hypothesis) 
losses.mean_absolute_percentage_error(target, hypothesis).numpy().mean()

target = [[0., 1.], [0., 0.]]
hypothesis = [[1., 1.], [1., 0.]]
cost = losses.MeanSquaredError()
cost(target, hypothesis) 
losses.mean_squared_error(target, hypothesis).numpy().mean()

target = [[0., 1.], [0., 0.]]
hypothesis = [[1., 1.], [1., 0.]]
cost = losses.MeanSquaredLogarithmicError()
cost(target, hypothesis) 
losses.mean_squared_logarithmic_error(target, hypothesis).numpy().mean()

target = [[0., 1.], [0., 0.]]
hypothesis = [[1., 1.], [0., 0.]]
cost = losses.Poisson()
cost(target, hypothesis) 
losses.poisson(target, hypothesis).numpy().mean()

target = [[0., 1.], [0., 0.]]
hypothesis = [[0.6, 0.4], [0.4, 0.6]]
cost = losses.SquaredHinge()
cost(target, hypothesis) 
losses.squared_hinge(target, hypothesis).numpy().mean()
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


### Profiler
- https://www.tensorflow.org/guide/profiler
```python

```



## Reference
- https://www.tensorflow.org/guide
- https://www.tensorflow.org/tutorials
- https://www.tensorflow.org/resources/models-datasets
- https://www.tensorflow.org/resources/libraries-extensions
- https://github.com/deeplearningzerotoall/TensorFlow/tree/master/tf_2.x
    
### APIs
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
- https://www.tensorflow.org/api_docs/python/tf/keras/callbacks

