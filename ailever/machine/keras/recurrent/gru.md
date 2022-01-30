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
import tensorflow as tf
from tensorflow.keras import layers

# [TimeMajor](sequence, batch, feature)
x = tf.random.normal(shape=(2, 32, 8))                                                 # x.shape                # (2, 32, 8) 
cell0_h = tf.random.normal(shape=(32, 4))                                                 # h.shape                # (32, 4) 

layer = layers.GRU(
    units=4, activation='tanh', recurrent_activation='sigmoid', 
    use_bias=True,
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', reset_after=True,
    stateful=False, time_major=True, unroll=False, 
    return_sequences=True, return_state=True)
cells_h_, cell2_h_ = layer(x, initial_state=[cell0_h])                                                          # layer.weights[0].shape  # (8, 12)
                                                                                                                # layer.weights[1].shape  # (4, 12)
                                                                                                                # layer.weights[2].shape  # (2, 12)
cell_W_z = layer.weights[0][:, 0:4]                                                                             # W_z.shape               # (8,4)
cell_W_r = layer.weights[0][:, 4:8]                                                                             # W_r.shape               # (8,4)
cell_W_c = layer.weights[0][:, 8:12]                                                                            # W_c.shape               # (8,4)
cell_U_z = layer.weights[1][:, 0:4]                                                                             # U_z.shape               # (4,4)
cell_U_r = layer.weights[1][:, 4:8]                                                                             # U_r.shape               # (4,4)
cell_U_c = layer.weights[1][:, 8:12]                                                                            # U_c.shape               # (4,4)
cell_b0_z = layer.weights[2][0, 0:4]                                                                            # b0_z.shape              # (, 4)
cell_b0_r = layer.weights[2][0, 4:8]                                                                            # b0_r.shape              # (, 4)
cell_b0_c = layer.weights[2][0, 8:12]                                                                           # b0_c.shape              # (, 4)
cell_b1_z = layer.weights[2][1, 0:4]                                                                            # b1_z.shape              # (, 4)
cell_b1_r = layer.weights[2][1, 4:8]                                                                            # b1_r.shape              # (, 4)
cell_b1_c = layer.weights[2][1, 8:12]                                                                           # b1_c.shape              # (, 4)



# [CELL1 Operation]
cell1_xW_z = tf.einsum('ij,jk->ik', x[0, :, :], cell_W_z)                                                       # cell1_xW_z.shape              # (32, 4)
cell1_hU_z = tf.einsum('ij,jk->ik', cell0_h, cell_U_z)                                                          # cell1_hU_z.shape              # (32, 4)
cell1_xW_r = tf.einsum('ij,jk->ik', x[0, :, :], cell_W_r)                                                       # cell1_xW_r.shape              # (32, 4)
cell1_hU_r = tf.einsum('ij,jk->ik', cell0_h, cell_U_r)                                                          # cell1_hU_r.shape              # (32, 4)
cell1_z = tf.sigmoid((cell1_xW_z + cell_b0_z) + (cell1_hU_z + cell_b1_z))                                       # cell1_z.shape                 # (32, 4)
cell1_r = tf.sigmoid((cell1_xW_r + cell_b0_r) + (cell1_hU_r + cell_b1_r))                                       # cell1_r.shape                 # (32, 4)
cell1_xW_c = tf.einsum('ij,jk->ik', x[0, :, :], cell_W_c)                                                       # cell1_xW_c.shape              # (32, 4)
cell1_transformed_hU_c = tf.einsum('ij,ij->ij', cell1_r, tf.einsum('ij,jk->ik', cell0_h, cell_U_c) + cell_b1_c) # cell1_transformed_hU_c.shape  # (32, 4)
cell1_c = tf.tanh((cell1_xW_c + cell_b0_c) + cell1_transformed_hU_c)                                            # cell1_c.shape                 # (32, 4)
cell1_h = tf.einsum('ij,ij->ij', (1-cell1_z), cell1_c) + tf.einsum('ij,ij->ij', cell1_z, cell0_h)

# [CELL2 Operation]
cell2_xW_z = tf.einsum('ij,jk->ik', x[1, :, :], cell_W_z)                                                       # cell2_xW_z.shape              # (32, 4)
cell2_hU_z = tf.einsum('ij,jk->ik', cell1_h, cell_U_z)                                                          # cell2_hU_z.shape              # (32, 4)
cell2_xW_r = tf.einsum('ij,jk->ik', x[1, :, :], cell_W_r)                                                       # cell2_xW_r.shape              # (32, 4)
cell2_hU_r = tf.einsum('ij,jk->ik', cell1_h, cell_U_r)                                                          # cell2_hU_r.shape              # (32, 4)
cell2_z = tf.sigmoid((cell2_xW_z + cell_b0_z) + (cell2_hU_z + cell_b1_z))                                       # cell2_z.shape                 # (32, 4)
cell2_r = tf.sigmoid((cell2_xW_r + cell_b0_r) + (cell2_hU_r + cell_b1_r))                                       # cell2_r.shape                 # (32, 4)
cell2_xW_c = tf.einsum('ij,jk->ik', x[1, :, :], cell_W_c)                                                       # cell2_xW_c.shape              # (32, 4)
cell2_transformed_hU_c = tf.einsum('ij,ij->ij', cell2_r, tf.einsum('ij,jk->ik', cell1_h, cell_U_c) + cell_b1_c) # cell2_transformed_hU_c.shape  # (32, 4)
cell2_c = tf.tanh((cell2_xW_c + cell_b0_c) + cell2_transformed_hU_c)                                            # cell2_c.shape                 # (32, 4)
cell2_h = tf.einsum('ij,ij->ij', (1-cell2_z), cell2_c) + tf.einsum('ij,ij->ij', cell2_z, cell1_h)

cell2_h - cell2_h_
```

`reset_after=False (tf version1 default)`, `time_major=False`
```python
```

`reset_after=False (tf version1 default)`, `time_major=True`
```python
```

### Argument: stateful

