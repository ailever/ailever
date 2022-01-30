## GRU(Gated Recurrent Unit)
- https://arxiv.org/pdf/1412.3555.pdf
- https://en.wikipedia.org/wiki/Gated_recurrent_unit

![image](https://user-images.githubusercontent.com/56889151/151705543-e715775d-282b-42fc-a7a6-6dd48c4ebf51.png)

### From GRUCell
`time_major=False`
```python
import tensorflow as tf
from tensorflow.keras import layers

# [BatchFirst](batch, sequence, feature)
x = tf.random.normal(shape=(32, 2, 8))                                              # x.shape                # (32, 2, 8) 
h = tf.random.normal(shape=(32, 4))                                                 # h.shape                # (32, 4) 

cell = layers.GRUCell(
    units=4, activation='tanh', recurrent_activation='sigmoid', 
    use_bias=True,
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', reset_after=True)
layer = layers.RNN(cell, stateful=False, time_major=False, unroll=False, return_sequences=True, return_state=True) 
cells_h_, cell2_h_ = layer(x, initial_state=[cell0_h])

# [CELL1 Operation]
cell1_x = x[:, 0, :]
cell1_h, (cell1_h, ) = cell(cell1_x, states=[cell0_h])

# [CELL2 Operation]
cell2_x = x[:, 1, :]
cell2_h, (cell2_h) = cell(cell2_x, states=[cell1_h])

cell2_h - cell2_h_, cell2_c - cell2_c_
```

`time_major=True`
```python
import tensorflow as tf
from tensorflow.keras import layers

# [TimeMajor](sequence, batch, feature)
x = tf.random.normal(shape=(2, 32, 8))                                              # x.shape                # (2, 32, 8) 
h = tf.random.normal(shape=(32, 4))                                                 # h.shape                # (32, 4) 

cell = layers.GRUCell(
    units=4, activation='tanh', recurrent_activation='sigmoid', 
    use_bias=True,
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', reset_after=True)
layer = layers.RNN(cell, stateful=False, time_major=True, unroll=False, return_sequences=True, return_state=True) 
cells_h_, cell2_h_ = layer(x, initial_state=[cell0_h])

# [CELL1 Operation]
cell1_x = x[0, :, :]
cell1_h, (cell1_h, ) = cell(cell1_x, states=[cell0_h])

# [CELL2 Operation]
cell2_x = x[1, :, :]
cell2_h, (cell2_h) = cell(cell2_x, states=[cell1_h])

cell2_h - cell2_h_, cell2_c - cell2_c_
```



### Argument: time_major
`reset_after=True (tf version2 default)`, `time_major=False`
```python
```

`reset_after=True (tf version2 default)`, `time_major=True`
```python
```

`reset_after=False (tf version1 default)`, `time_major=False`
```python
```

`reset_after=False (tf version1 default)`, `time_major=True`
```python
```

### Argument: stateful

