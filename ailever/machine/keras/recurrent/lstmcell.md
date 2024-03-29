## LSTM(Long Short-Term Memory) Cell
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTMCell
- https://en.wikipedia.org/wiki/Long_short-term_memory
![image](https://user-images.githubusercontent.com/56889151/151692355-f7cb33fc-4e81-4a62-a353-53031097a616.png)

```python
import tensorflow as tf
from tensorflow.keras import layers

# (batch, feature)
x = tf.random.normal(shape=(32, 8))              # x.shape               # (32, 8) 
h = tf.random.normal(shape=(32, 4))              # h.shape               # (32, 4) 
c = tf.random.normal(shape=(32, 4))              # c.shape               # (32, 4)

cell = layers.LSTMCell(
    units=4, activation='tanh', recurrent_activation='sigmoid', 
    use_bias=True, unit_forget_bias=True,  
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal')
h_, (h_, c_) = cell(x, states=[h, c])            # cell.weights[0].shape # (8, 16)
                                                 # cell.weights[1].shape # (4, 16)
                                                 # cell.weights[2].shape # (, 16)

xW = tf.einsum('ij,jk->ik', x, cell.weights[0])  # xW.shape              # (32, 16)
hW = tf.einsum('ij,jk->ik', h, cell.weights[1])  # hW.shape              # (32, 16)
b = cell.weights[2]                              # b.shape               # (,16)
bilinear = xW + hW + b                           # bilinear.shape        # (32, 16)

i = tf.sigmoid(bilinear[:, 0:4])                 # i.shape               # (32, 4)
f = tf.sigmoid(bilinear[:, 4:8])                 # f.shape               # (32, 4)
g = tf.tanh(bilinear[:, 8:12])                   # g.shape               # (32, 4)
o = tf.sigmoid(bilinear[:, 12:16])               # o.shape               # (32, 4)
c = tf.einsum('ij,ij->ij', f, c) + tf.einsum('ij,ij->ij', i, g)
h = tf.einsum('ij,ij->ij', o, tf.tanh(c))

h - h_, c - c_
```


### From LSTM

`time_major=False`
```python
import tensorflow as tf
from tensorflow.keras import layers

# [BatchFirst](batch, 1, feature)
x = tf.random.normal(shape=(32, 1, 8))                          # x.shape                     # (32, 1, 8) 
h = tf.random.normal(shape=(32, 4))                             # h.shape                     # (32, 4) 
c = tf.random.normal(shape=(32, 4))                             # c.shape                     # (32, 4)

layer = layers.LSTM(
    units=4, activation='tanh', recurrent_activation='sigmoid', 
    use_bias=True, unit_forget_bias=True,  
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    stateful=False, time_major=False, unroll=False,
    return_sequences=True, return_state=True)
h_, (h_, c_) = layer.cell(x[:, 0, :], states=[h, c])            # layer.cell.weights[0].shape # (8, 16)
                                                                # layer.cell.weights[1].shape # (4, 16)
                                                                # layer.cell.weights[2].shape # (, 16)

xW = tf.einsum('ij,jk->ik', x[:, 0, :], layer.cell.weights[0])  # xW.shape                    # (32, 16)
hW = tf.einsum('ij,jk->ik', h, layer.cell.weights[1])           # hW.shape                    # (32, 16)
b = layer.cell.weights[2]                                       # b.shape                     # (,16)
bilinear = xW + hW + b                                          # bilinear.shape              # (32, 16)

i = tf.sigmoid(bilinear[:, 0:4])                                # i.shape                     # (32, 4)
f = tf.sigmoid(bilinear[:, 4:8])                                # f.shape                     # (32, 4)
g = tf.tanh(bilinear[:, 8:12])                                  # g.shape                     # (32, 4)
o = tf.sigmoid(bilinear[:, 12:16])                              # o.shape                     # (32, 4)
c = tf.einsum('ij,ij->ij', f, c) + tf.einsum('ij,ij->ij', i, g)
h = tf.einsum('ij,ij->ij', o, tf.tanh(c))

h - h_, c - c_
```


`time_major=True`
```python
import tensorflow as tf
from tensorflow.keras import layers

# [TimeMajor](1, batch, feature)
x = tf.random.normal(shape=(1, 32, 8))                          # x.shape                     # (1, 32, 8) 
h = tf.random.normal(shape=(32, 4))                             # h.shape                     # (32, 4) 
c = tf.random.normal(shape=(32, 4))                             # c.shape                     # (32, 4)

layer = layers.LSTM(
    units=4, activation='tanh', recurrent_activation='sigmoid', 
    use_bias=True, unit_forget_bias=True,  
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    stateful=False, time_major=True, unroll=False,
    return_sequences=True, return_state=True)
h_, (h_, c_) = layer.cell(x[0, :, :], states=[h, c])            # layer.cell.weights[0].shape # (8, 16)
                                                                # layer.cell.weights[1].shape # (4, 16)
                                                                # layer.cell.weights[2].shape # (, 16)

xW = tf.einsum('ij,jk->ik', x[0, :, :], layer.cell.weights[0])  # xW.shape                    # (32, 16)
hW = tf.einsum('ij,jk->ik', h, layer.cell.weights[1])           # hW.shape                    # (32, 16)
b = layer.cell.weights[2]                                       # b.shape                     # (,16)
bilinear = xW + hW + b                                          # bilinear.shape              # (32, 16)

i = tf.sigmoid(bilinear[:, 0:4])                                # i.shape                     # (32, 4)
f = tf.sigmoid(bilinear[:, 4:8])                                # f.shape                     # (32, 4)
g = tf.tanh(bilinear[:, 8:12])                                  # g.shape                     # (32, 4)
o = tf.sigmoid(bilinear[:, 12:16])                              # o.shape                     # (32, 4)
c = tf.einsum('ij,ij->ij', f, c) + tf.einsum('ij,ij->ij', i, g)
h = tf.einsum('ij,ij->ij', o, tf.tanh(c))

h - h_, c - c_
```
