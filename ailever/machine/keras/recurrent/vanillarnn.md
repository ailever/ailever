## Vanilla RNN(Recurrent Neural Network)
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
### From SimpleRNNCell
`time_major=False`
```python
import tensorflow as tf
from tensorflow.keras import layers

# [BatchFirst](batch, sequence, feature)
x = tf.random.normal(shape=(32, 2, 8))                        # x.shape               # (32, 2, 8) 
cell0_h = tf.random.normal(shape=(32, 4))                     # cell0_h.shape         # (32, 4) 

cell = layers.SimpleRNNCell(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal')
layer = layers.RNN(cell, stateful=False, time_major=False, unroll=False, return_sequences=True, return_state=True) 
cells_h_, cell2_h_ = layer(x, initial_state=[cell0_h])        # layer.weights[0].shape # (8, 4)
                                                              # layer.weights[1].shape # (4, 4)
                                                              # layer.weights[2].shape # (, 4)

# [CELL1 Operation]
cell1_x = x[:, 0, :]
cell1_h, (cell1_h, ) = cell(cell1_x, states=[cell0_h])

# [CELL2 Operation]
cell2_x = x[:, 1, :]
cell2_h, (cell2_h, ) = cell(cell2_x, states=[cell1_h])

cell2_h - cell2_h_
```
`time_major=True`
```python
import tensorflow as tf
from tensorflow.keras import layers

# [TimeMajor](sequence, batch, feature)
x = tf.random.normal(shape=(2, 32, 8))                        # x.shape               # (2, 32, 8) 
cell0_h = tf.random.normal(shape=(32, 4))                     # cell0_h.shape         # (32, 4) 

cell = layers.SimpleRNNCell(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal')
layer = layers.RNN(cell, stateful=False, time_major=True, unroll=False, return_sequences=True, return_state=True) 
cells_h_, cell2_h_ = layer(x, initial_state=[cell0_h])        # layer.weights[0].shape # (8, 4)
                                                              # layer.weights[1].shape # (4, 4)
                                                              # layer.weights[2].shape # (, 4)

# [CELL1 Operation]
cell1_x = x[0, :, :]
cell1_h, (cell1_h, ) = cell(cell1_x, states=[cell0_h])

# [CELL2 Operation]
cell2_x = x[1, :, :]
cell2_h, (cell2_h, ) = cell(cell2_x, states=[cell1_h])

cell2_h - cell2_h_
```

### Argument: time_major
`time_major=False`
```python
import tensorflow as tf
from tensorflow.keras import layers

# [BatchFirst](batch, sequence, feature)
x = tf.random.normal(shape=(32, 2, 8))                           # x.shape               # (32, 2, 8) 
cell0_h = tf.random.normal(shape=(32, 4))                        # cell0_h.shape         # (32, 4) 

layer = layers.SimpleRNN(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    stateful=False, time_major=False, unroll=False,
    return_sequences=True, return_state=True)
cells_h_, cell2_h_ = layer(x, initial_state=[cell0_h])           # layer.weights[0].shape # (8, 4)
                                                                 # layer.weights[1].shape # (4, 4)
                                                                 # layer.weights[2].shape # (, 4)


        
# [CELL1 Operation]        
cell1_xW = tf.einsum('ij,jk->ik', x[:, 0, :], layer.weights[0])  # cell1_xW.shape              # (32, 4)
cell1_hW = tf.einsum('ij,jk->ik', cell0_h, layer.weights[1])     # cell1_hW.shape              # (32, 4)
cell_b = layer.weights[2]                                        # cell_b.shape                # (, 4)
cell1_bilinear = cell1_xW + cell1_hW + cell_b                    # cell1_bilinear.shape        # (32, 4)
cell1_h = tf.tanh(cell1_bilinear)                                # cell1_h.shape               # (32, 4)

# [CELL2 Operation]
cell2_xW = tf.einsum('ij,jk->ik', x[:, 1, :], layer.weights[0])  # cell2_xW.shape              # (32, 4)
cell2_hW = tf.einsum('ij,jk->ik', cell1_h, layer.weights[1])     # cell2_hW.shape              # (32, 4)
cell_b = layer.weights[2]                                        # cell_b.shape                # (, 4)
cell2_bilinear = cell2_xW + cell2_hW + cell_b                    # cell2_bilinear.shape        # (32, 4)
cell2_h = tf.tanh(cell2_bilinear)                                # cell2_h.shape               # (32, 4)

cell2_h - cell2_h_
```

```python
import tensorflow as tf
from tensorflow.keras import layers

# [BatchFirst](batch, sequence, feature)
x = tf.random.normal(shape=(32, 2, 8))                           # x.shape               # (32, 2, 8) 
cell0_h = tf.random.normal(shape=(32, 4))                        # cell0_h.shape         # (32, 4) 

layer = layers.SimpleRNN(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    stateful=False, time_major=False, unroll=False,
    return_sequences=True, return_state=True)
cells_h_, cell2_h_ = layer(x, initial_state=[cell0_h])           # layer.weights[0].shape # (8, 4)
                                                                 # layer.weights[1].shape # (4, 4)
                                                                 # layer.weights[2].shape # (, 4)


        
# [CELL1 Operation]
cell1_x = x[:, 0, :]
cell1_h, (cell1_h, ) = layer.cell(cell1_x, states=[cell0_h])

# [CELL2 Operation]
cell2_x = x[:, 1, :]
cell2_h, (cell2_h, ) = layer.cell(cell2_x, states=[cell1_h])

cell2_h - cell2_h_
```



`time_major=True`
```python
import tensorflow as tf
from tensorflow.keras import layers

# [TimeMajor](sequence, batch, feature)
x = tf.random.normal(shape=(2, 32, 8))                           # x.shape               # (2, 32, 8) 
cell0_h = tf.random.normal(shape=(32, 4))                        # cell0_h.shape         # (32, 4) 

layer = layers.SimpleRNN(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    stateful=False, time_major=True, unroll=False,
    return_sequences=True, return_state=True)
cells_h_, cell2_h_ = layer(x, initial_state=[cell0_h])           # layer.weights[0].shape # (8, 4)
                                                                 # layer.weights[1].shape # (4, 4)
                                                                 # layer.weights[2].shape # (, 4)


        
# [CELL1 Operation]        
cell1_xW = tf.einsum('ij,jk->ik', x[0, :, :], layer.weights[0])  # cell1_xW.shape              # (32, 4)
cell1_hW = tf.einsum('ij,jk->ik', cell0_h, layer.weights[1])     # cell1_hW.shape              # (32, 4)
cell_b = layer.weights[2]                                        # cell_b.shape                # (, 4)
cell1_bilinear = cell1_xW + cell1_hW + cell_b                    # cell1_bilinear.shape        # (32, 4)
cell1_h = tf.tanh(cell1_bilinear)                                # cell1_h.shape               # (32, 4)

# [CELL2 Operation]
cell2_xW = tf.einsum('ij,jk->ik', x[1, :, :], layer.weights[0])  # cell2_xW.shape              # (32, 4)
cell2_hW = tf.einsum('ij,jk->ik', cell1_h, layer.weights[1])     # cell2_hW.shape              # (32, 4)
cell_b = layer.weights[2]                                        # cell_b.shape                # (, 4)
cell2_bilinear = cell2_xW + cell2_hW + cell_b                    # cell2_bilinear.shape        # (32, 4)
cell2_h = tf.tanh(cell2_bilinear)                                # cell2_h.shape               # (32, 4)

cell2_h - cell2_h_
```

```python
import tensorflow as tf
from tensorflow.keras import layers

# [TimeMajor](sequence, batch, feature)
x = tf.random.normal(shape=(2, 32, 8))                           # x.shape               # (2, 32, 8) 
cell0_h = tf.random.normal(shape=(32, 4))                        # cell0_h.shape         # (32, 4) 

layer = layers.SimpleRNN(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal',
    stateful=False, time_major=True, unroll=False,
    return_sequences=True, return_state=True)
cells_h_, cell2_h_ = layer(x, initial_state=[cell0_h])           # layer.weights[0].shape # (8, 4)
                                                                 # layer.weights[1].shape # (4, 4)
                                                                 # layer.weights[2].shape # (, 4)


        
# [CELL1 Operation]
cell1_x = x[0, :, :]
cell1_h, (cell1_h, ) = layer.cell(cell1_x, states=[cell0_h])

# [CELL2 Operation]
cell2_x = x[1, :, :]
cell2_h, (cell2_h, ) = layer.cell(cell2_x, states=[cell1_h])

cell2_h - cell2_h_
```


### Argument: stateful
