## Vanilla RNN(Recurrent Neural Network) Cell
- https://en.wikipedia.org/wiki/Recurrent_neural_network
![image](https://user-images.githubusercontent.com/56889151/151692305-7a58363c-b9a2-483b-a814-2f35a90780ea.png)

```python
import tensorflow as tf
from tensorflow.keras import layers

# (batch, feature)
x = tf.random.normal(shape=(32, 8))              # x.shape               # (32, 8) 
h = tf.random.normal(shape=(32, 4))              # h.shape               # (32, 4) 

cell = layers.SimpleRNNCell(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal')
h_, (h_, ) = cell(x, states=[h])                 # cell.weights[0].shape # (8, 4)
                                                 # cell.weights[1].shape # (4, 4)
                                                 # cell.weights[2].shape # (, 4)

xW = tf.einsum('ij,jk->ik', x, cell.weights[0])  # xW.shape              # (32, 4)
hW = tf.einsum('ij,jk->ik', h, cell.weights[1])  # hW.shape              # (32, 4)
b = cell.weights[2]                              # b.shape               # (, 4)
bilinear = xW + hW + b                           # bilinear.shape        # (32, 4)
h = tf.tanh(bilinear)                            # h.shape               # (32, 4)

h - h_
```

`time_major=False`
```python
import tensorflow as tf
from tensorflow.keras import layers

# [BatchFirst](batch, 1, feature)
x = tf.random.normal(shape=(32, 1, 8))                           # x.shape                    # (32, 1, 8) 
h = tf.random.normal(shape=(32, 4))                              # h.shape                    # (32, 4) 

layer = layers.SimpleRNN(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    stateful=False, time_major=False, unroll=False,
    return_sequences=True, return_state=True)
h_, (h_, ) = layer.cell(x[:, 0, :], states=[h])                 # layer.cell.weights[0].shape # (8, 4)
                                                                # layer.cell.weights[1].shape # (4, 4)
                                                                # layer.cell.weights[2].shape # (, 4)
        
xW = tf.einsum('ij,jk->ik', x[:, 0, :], layer.cell.weights[0])  # xW.shape                    # (32, 4)
hW = tf.einsum('ij,jk->ik', h, layer.cell.weights[1])           # hW.shape                    # (32, 4)
b = layer.cell.weights[2]                                       # b.shape                     # (, 4)
bilinear = xW + hW + b                                          # bilinear.shape              # (32, 4)
h = tf.tanh(bilinear)                                           # h.shape                     # (32, 4)

h - h_
```

`time_major=True`
```python

```
