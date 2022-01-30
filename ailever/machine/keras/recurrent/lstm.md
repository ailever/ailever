## LSTM
- https://en.wikipedia.org/wiki/Long_short-term_memory  

![image](https://user-images.githubusercontent.com/56889151/151700189-52624f1f-9e25-4fbd-adc5-b1919fcf6895.png)


### From LSTMCell
```python
```


### Argument: time_major
`time_major=False`
```python
import tensorflow as tf
from tensorflow.keras import layers

# [BatchFirst](batch, sequence, feature) <------ # number of sequence = number of cell
x = tf.random.normal(shape=(32, 2, 8))           # x.shape       # (32, 2, 8) <-------- cell1 & cell2 
cell0_h = tf.random.normal(shape=(32, 4))        # cell0_h.shape # (32, 4) 
cell0_c = tf.random.normal(shape=(32, 4))        # cell0_c.shape # (32, 4)

layer = layers.LSTM(
    units=4, activation='tanh', recurrent_activation='sigmoid',
    use_bias=True, unit_forget_bias=True,  
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    stateful=False, time_major=False, unroll=False, # [batch major(batch/time/feature)]
    return_sequences=True, return_state=True) 
cells_h_, cell2_h_, cell2_c_ = layer(x, initial_state=[cell0_h, cell0_c])
#layer.weights[0].shape # (8, 16)
#layer.weights[1].shape # (4, 16)
#layer.weights[0].shape # (, 16)
# cells_h_.shape # (32, 2, 4) --------> cells_h_[:, -1, :] = cell2_h_
                             #--------> cell1_h, (cell1_h, cell1_c) = LSTMCell1(x[:, 0, :], states=[cell0_h, cell0_c])
                             #--------> cell2_h, (cell2_h, cell2_c) = LSTMCell2(x[:, 1, :], states=[cell1_h, cell1_c])
# cell0_h.shape  # (32, 4)
# cell0_c.shape  # (32, 4)
# cell2_h_.shape # (32, 4)
# cell2_c_.shape # (32, 4)



# [CELL1 Operation]
cell1_x = x[:, 0, :]
cell1_xW = tf.einsum('ij,jk->ik', cell1_x, layer.weights[0]) # cell1_xW.shape       # (32, 16)
cell1_hW = tf.einsum('ij,jk->ik', cell0_h, layer.weights[1]) # cell1_hW.shape       # (32, 16)
cell1_b = layer.weights[2]                                   # cell1_b.shape        # (,16)
cell1_bilinear = cell1_xW + cell1_hW + cell1_b               # cell1_bilinear.shape # (32, 16)

cell1_i = tf.sigmoid(cell1_bilinear[:, 0:4])                 # cell1_i.shape        # (32, 4)
cell1_f = tf.sigmoid(cell1_bilinear[:, 4:8])                 # cell1_f.shape        # (32, 4)
cell1_g = tf.tanh(cell1_bilinear[:, 8:12])                   # cell1_g.shape        # (32, 4)
cell1_o = tf.sigmoid(cell1_bilinear[:, 12:16])               # cell1_o.shape        # (32, 4)
cell1_c = tf.einsum('ij,ij->ij', cell1_f, cell0_c) + tf.einsum('ij,ij->ij', cell1_i, cell1_g)
cell1_h = tf.einsum('ij,ij->ij', cell1_o, tf.tanh(cell1_c))

# [CELL2 Operation]
cell2_x = x[:, 1, :]
cell2_xW = tf.einsum('ij,jk->ik', cell2_x, layer.weights[0]) # cell2_xW.shape       # (32, 16)
cell2_hW = tf.einsum('ij,jk->ik', cell1_h, layer.weights[1]) # cell2_hW.shape       # (32, 16)
cell2_b = layer.weights[2]                                   # cell2_b.shape        # (,16)
cell2_bilinear = cell2_xW + cell2_hW + cell2_b               # cell2_bilinear.shape # (32, 16)

cell2_i = tf.sigmoid(cell2_bilinear[:, 0:4])                 # cell2_i.shape        # (32, 4)
cell2_f = tf.sigmoid(cell2_bilinear[:, 4:8])                 # cell2_f.shape        # (32, 4)
cell2_g = tf.tanh(cell2_bilinear[:, 8:12])                   # cell2_g.shape        # (32, 4)
cell2_o = tf.sigmoid(cell2_bilinear[:, 12:16])               # cell2_o.shape        # (32, 4)
cell2_c = tf.einsum('ij,ij->ij', cell2_f, cell1_c) + tf.einsum('ij,ij->ij', cell2_i, cell2_g)
cell2_h = tf.einsum('ij,ij->ij', cell2_o, tf.tanh(cell2_c))

cell2_h - cell2_h_ , cell2_c - cell2_c_
```
```python
import tensorflow as tf
from tensorflow.keras import layers

# [BatchFirst](batch, sequence, feature) <------ # number of sequence = number of cell
x = tf.random.normal(shape=(32, 2, 8))           # x.shape       # (32, 2, 8) <-------- cell1 & cell2 
cell0_h = tf.random.normal(shape=(32, 4))        # cell0_h.shape # (32, 4) 
cell0_c = tf.random.normal(shape=(32, 4))        # cell0_c.shape # (32, 4)

layer = layers.LSTM(
    units=4, activation='tanh', recurrent_activation='sigmoid',
    use_bias=True, unit_forget_bias=True,  
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    stateful=False, time_major=False, unroll=False, # [batch major(batch/time/feature)]
    return_sequences=True, return_state=True) 
cells_h_, cell2_h_, cell2_c_ = layer(x, initial_state=[cell0_h, cell0_c])
#layer.weights[0].shape # (8, 16)
#layer.weights[1].shape # (4, 16)
#layer.weights[0].shape # (, 16)
# cells_h_.shape # (32, 2, 4) --------> cells_h_[:, -1, :] = cell2_h_
                             #--------> cell1_h, (cell1_h, cell1_c) = LSTMCell1(x[:, 0, :], states=[cell0_h, cell0_c])
                             #--------> cell2_h, (cell2_h, cell2_c) = LSTMCell2(x[:, 1, :], states=[cell1_h, cell1_c])
# cell0_h.shape  # (32, 4)
# cell0_c.shape  # (32, 4)
# cell2_h_.shape # (32, 4)
# cell2_c_.shape # (32, 4)



# [CELL1 Operation]
cell1_x = x[:, 0, :]
cell1_h, (cell1_h, cell1_c) = layer.cell(cell1_x, states=[cell0_h, cell0_c])

# [CELL2 Operation]
cell2_x = x[:, 1, :]
cell2_h, (cell2_h, cell2_c) = layer.cell(cell2_x, states=[cell1_h, cell1_c])

cell2_h - cell2_h_ , cell2_c - cell2_c_
```


`time_major=True`
```python
import tensorflow as tf
from tensorflow.keras import layers

# [TimeMajor](sequence, batch, feature) <------- # number of sequence = number of cell
x = tf.random.normal(shape=(2, 32, 8))           # x.shape       # (2, 32, 8) <-------- cell1 & cell2 
cell0_h = tf.random.normal(shape=(32, 4))        # cell0_h.shape # (32, 4) 
cell0_c = tf.random.normal(shape=(32, 4))        # cell0_c.shape # (32, 4)

layer = layers.LSTM(
    units=4, activation='tanh', recurrent_activation='sigmoid',
    use_bias=True, unit_forget_bias=True,  
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    stateful=False, time_major=True, unroll=False, # [time major(time/batch/feature)]
    return_sequences=True, return_state=True) 
cells_h_, cell2_h_, cell2_c_ = layer(x, initial_state=[cell0_h, cell0_c])
#layer.weights[0].shape # (8, 16)
#layer.weights[1].shape # (4, 16)
#layer.weights[0].shape # (, 16)
# cells_h_.shape # (2, 32, 4)  --------> cells_h_[-1, :, :] = cell2_h_
                              #--------> cell1_h, (cell1_h, cell1_c) = LSTMCell1(x_[0, :, :], states=[cell0_h, cell0_c])
                              #--------> cell2_h, (cell2_h, cell2_c) = LSTMCell2(x_[1, :, :], states=[cell1_h, cell1_c])
# cell0_h.shape  # (32, 4)
# cell0_c.shape  # (32, 4)
# cell2_h_.shape # (32, 4)
# cell2_c_.shape # (32, 4)



# [CELL1 Operation]
cell1_x = x[0, :, :]
cell1_xW = tf.einsum('ij,jk->ik', cell1_x, layer.weights[0]) # cell1_xW.shape       # (32, 16)
cell1_hW = tf.einsum('ij,jk->ik', cell0_h, layer.weights[1]) # cell1_hW.shape       # (32, 16)
cell1_b = layer.weights[2]                                   # cell1_b.shape        # (,16)
cell1_bilinear = cell1_xW + cell1_hW + cell1_b               # cell1_bilinear.shape # (32, 16)

cell1_i = tf.sigmoid(cell1_bilinear[:, 0:4])                 # cell1_i.shape        # (32, 4)
cell1_f = tf.sigmoid(cell1_bilinear[:, 4:8])                 # cell1_f.shape        # (32, 4)
cell1_g = tf.tanh(cell1_bilinear[:, 8:12])                   # cell1_g.shape        # (32, 4)
cell1_o = tf.sigmoid(cell1_bilinear[:, 12:16])               # cell1_o.shape        # (32, 4)
cell1_c = tf.einsum('ij,ij->ij', cell1_f, cell0_c) + tf.einsum('ij,ij->ij', cell1_i, cell1_g)
cell1_h = tf.einsum('ij,ij->ij', cell1_o, tf.tanh(cell1_c))

# [CELL2 Operation]
cell2_x = x[1, :, :]
cell2_xW = tf.einsum('ij,jk->ik', cell2_x, layer.weights[0]) # cell2_xW.shape       # (32, 16)
cell2_hW = tf.einsum('ij,jk->ik', cell1_h, layer.weights[1]) # cell2_hW.shape       # (32, 16)
cell2_b = layer.weights[2]                                   # cell2_b.shape        # (,16)
cell2_bilinear = cell2_xW + cell2_hW + cell2_b               # cell2_bilinear.shape # (32, 16)

cell2_i = tf.sigmoid(cell2_bilinear[:, 0:4])                 # cell2_i.shape        # (32, 4)
cell2_f = tf.sigmoid(cell2_bilinear[:, 4:8])                 # cell2_f.shape        # (32, 4)
cell2_g = tf.tanh(cell2_bilinear[:, 8:12])                   # cell2_g.shape        # (32, 4)
cell2_o = tf.sigmoid(cell2_bilinear[:, 12:16])               # cell2_o.shape        # (32, 4)
cell2_c = tf.einsum('ij,ij->ij', cell2_f, cell1_c) + tf.einsum('ij,ij->ij', cell2_i, cell2_g)
cell2_h = tf.einsum('ij,ij->ij', cell2_o, tf.tanh(cell2_c))

cell2_h - cell2_h_ , cell2_c - cell2_c_
```
```python
import tensorflow as tf
from tensorflow.keras import layers

# [TimeMajor](sequence, batch, feature) <------- # number of sequence = number of cell
x = tf.random.normal(shape=(2, 32, 8))           # x.shape       # (2, 32, 8) <-------- cell1 & cell2 
cell0_h = tf.random.normal(shape=(32, 4))        # cell0_h.shape # (32, 4) 
cell0_c = tf.random.normal(shape=(32, 4))        # cell0_c.shape # (32, 4)

layer = layers.LSTM(
    units=4, activation='tanh', recurrent_activation='sigmoid',
    use_bias=True, unit_forget_bias=True,  
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    stateful=False, time_major=True, unroll=False, # [time major(time/batch/feature)]
    return_sequences=True, return_state=True) 
cells_h_, cell2_h_, cell2_c_ = layer(x, initial_state=[cell0_h, cell0_c])
#layer.weights[0].shape # (8, 16)
#layer.weights[1].shape # (4, 16)
#layer.weights[0].shape # (, 16)
# cells_h_.shape # (2, 32, 4)  --------> cells_h_[-1, :, :] = cell2_h_
                              #--------> cell1_h, (cell1_h, cell1_c) = LSTMCell1(x_[0, :, :], states=[cell0_h, cell0_c])
                              #--------> cell2_h, (cell2_h, cell2_c) = LSTMCell2(x_[1, :, :], states=[cell1_h, cell1_c])
# cell0_h.shape  # (32, 4)
# cell0_c.shape  # (32, 4)
# cell2_h_.shape # (32, 4)
# cell2_c_.shape # (32, 4)



# [CELL1 Operation]
cell1_x = x[0, :, :]
cell1_h, (cell1_h, cell1_c) = layer.cell(cell1_x, states=[cell0_h, cell0_c])

# [CELL2 Operation]
cell2_x = x[1, :, :]
cell2_h, (cell2_h, cell2_c) = layer.cell(cell2_x, states=[cell1_h, cell1_c])

cell2_h - cell2_h_ , cell2_c - cell2_c_
```



### Argument: stateful






