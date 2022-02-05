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

### Argument: go_backwards
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
    return_sequences=True, return_state=True, go_backwards=True)
cells_h_, cell2_h_ = layer(x, initial_state=[cell0_h])           # layer.weights[0].shape # (8, 4)
                                                                 # layer.weights[1].shape # (4, 4)
                                                                 # layer.weights[2].shape # (, 4)


        
# [CELL1 Operation]
cell1_x = x[:, 1, :]
cell1_h, (cell1_h, ) = layer.cell(cell1_x, states=[cell0_h])

# [CELL2 Operation]
cell2_x = x[:, 0, :]
cell2_h, (cell2_h, ) = layer.cell(cell2_x, states=[cell1_h])

cell2_h - cell2_h_
cells_h_[:,0,:] - cell1_h
cells_h_[:,1,:] - cell2_h
```

### Argument: stateful
`stateful=True`
```python
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras import layers, models, losses

# [BatchFirst](batch, sequence, feature)
batch_size = 16
train_x = tf.random.normal(shape=(32, 2, 8))                           # train_x.shape               # (32, 2, 8) 
train_y = tf.random.normal(shape=(32, 2, 4))                           # train_y.shape               # (32, 2, 4)
train_x_batch1 = train_x[:batch_size]                                  # train_x_batch1.shape        # (16, 2, 8)
train_x_batch2 = train_x[batch_size:batch_size*2]                      # train_x_batch2.shape        # (16, 2, 8)
train_y_batch1 = train_y[:batch_size]                                  # train_y_batch1.shape        # (16, 2, 4)
train_y_batch2 = train_y[batch_size:batch_size*2]                      # train_y_batch2.shape        # (16, 2, 4)
test_x_batch = tf.random.normal(shape=(16, 2, 8))                      # test_x_batch.shape          # (16, 2, 8) 
cell0_h = tf.random.normal(shape=(batch_size, 4))                      # cell0_h.shape               # (16, 4) 

# [Case1]: layer_ (= layer from case2)
layer_ = layers.SimpleRNN(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='ones', kernel_initializer='ones', recurrent_initializer='ones',
    stateful=True, time_major=False, unroll=False,
    return_sequences=True, return_state=True)
inputs = layers.Input(shape=(2, 8), batch_size=batch_size)
outputs = layer_(inputs, initial_state=[cell0_h])
model = models.Model(inputs, outputs[0])

# [Case1]: Train Full Batch
loss = losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
model.fit(train_x, train_y, batch_size=batch_size, epochs=1, shuffle=False)


# [Case2]: layer (= layer_ from case1)
layer = layers.SimpleRNN(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='ones', kernel_initializer='ones', recurrent_initializer='ones',
    stateful=True, time_major=False, unroll=False,
    return_sequences=True, return_state=True)
loss = losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# [Case2]: Train First Batch 
with tf.GradientTape() as tape:
    cells_h_after_batch1, cell2_h_after_batch1 = layer(train_x_batch1, initial_state=[cell0_h])
    cost = loss(train_y_batch1, cells_h_after_batch1)
gradients = tape.gradient(cost, layer.trainable_variables)
optimizer.apply_gradients(zip(gradients, layer.trainable_variables)) # optimizer.weights[-1], optimizer.get_weights(), optimizer.set_weights(optimizer.weights)

# [Case2]: Train Second Batch
with tf.GradientTape() as tape:
    cells_h_after_batch2, cell2_h_after_batch2 = layer(train_x_batch2, initial_state=[cell2_h_after_batch1])
    cost = loss(train_y_batch2, cells_h_after_batch2)
gradients = tape.gradient(cost, layer.trainable_variables)
optimizer.apply_gradients(zip(gradients, layer.trainable_variables)) # optimizer.weights[-1], optimizer.get_weights(), optimizer.set_weights(optimizer.weights)



cells_h_for_testbatch_1, cell2_h_for_testbatch_1 = model.layers[1](test_x_batch, initial_state=[cell2_h_after_batch2]) # from case1
cells_h_for_testbatch_2, cell2_h_for_testbatch_2 = model.layers[1](test_x_batch)                                       # from case1
cells_h_for_testbatch_3 = model.predict(test_x_batch)                                                                  # from case1
cell2_h_for_testbatch_3 = cells_h_for_testbatch_3[:, -1, :]                                                            # from case1

cells_h_for_testbatch, cell2_h_for_testbatch = layer(test_x_batch, initial_state=[cell2_h_after_batch2])               # from case2
cell2_h_for_testbatch - cell2_h_for_testbatch_1
```

`stateful=False`
```python
from copy import deepcopy
import tensorflow as tf
from tensorflow.keras import layers, models, losses

# [BatchFirst](batch, sequence, feature)
batch_size = 16
train_x = tf.random.normal(shape=(32, 2, 8))                           # train_x.shape               # (32, 2, 8) 
train_y = tf.random.normal(shape=(32, 2, 4))                           # train_y.shape               # (32, 2, 4)
train_x_batch1 = train_x[:batch_size]                                  # train_x_batch1.shape        # (16, 2, 8)
train_x_batch2 = train_x[batch_size:batch_size*2]                      # train_x_batch2.shape        # (16, 2, 8)
train_y_batch1 = train_y[:batch_size]                                  # train_y_batch1.shape        # (16, 2, 4)
train_y_batch2 = train_y[batch_size:batch_size*2]                      # train_y_batch2.shape        # (16, 2, 4)
test_x_batch = tf.random.normal(shape=(16, 2, 8))                      # test_x_batch.shape          # (16, 2, 8) 
cell0_h = tf.random.normal(shape=(batch_size, 4))                      # cell0_h.shape               # (16, 4) 

# [Case1]: layer_ (= layer from case2)
layer_ = layers.SimpleRNN(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='ones', kernel_initializer='ones', recurrent_initializer='ones',
    stateful=False, time_major=False, unroll=False,
    return_sequences=True, return_state=True)
inputs = layers.Input(shape=(2, 8), batch_size=batch_size)
outputs = layer_(inputs, initial_state=[cell0_h])
model = models.Model(inputs, outputs[0])

# [Case1]: Train Full Batch
loss = losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.compile(optimizer=optimizer, loss=loss, metrics=["mae"])
model.fit(train_x, train_y, batch_size=batch_size, epochs=1, shuffle=False)


# [Case2]: layer (= layer_ from case1)
layer = layers.SimpleRNN(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='ones', kernel_initializer='ones', recurrent_initializer='ones',
    stateful=True, time_major=False, unroll=False,
    return_sequences=True, return_state=True)
loss = losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# [Case2]: Train First Batch 
with tf.GradientTape() as tape:
    cells_h_after_batch1, cell2_h_after_batch1 = layer(train_x_batch1, initial_state=[cell0_h])
    cost = loss(train_y_batch1, cells_h_after_batch1)
gradients = tape.gradient(cost, layer.trainable_variables)
optimizer.apply_gradients(zip(gradients, layer.trainable_variables)) # optimizer.weights[-1], optimizer.get_weights(), optimizer.set_weights(optimizer.weights)

# [Case2]: Train Second Batch
with tf.GradientTape() as tape:
    cells_h_after_batch2, cell2_h_after_batch2 = layer(train_x_batch2, initial_state=[cell2_h_after_batch1])
    cost = loss(train_y_batch2, cells_h_after_batch2)
gradients = tape.gradient(cost, layer.trainable_variables)
optimizer.apply_gradients(zip(gradients, layer.trainable_variables)) # optimizer.weights[-1], optimizer.get_weights(), optimizer.set_weights(optimizer.weights)



cells_h_for_testbatch_1, cell2_h_for_testbatch_1 = model.layers[1](test_x_batch, initial_state=[cell2_h_after_batch2]) # from case1
cells_h_for_testbatch_2, cell2_h_for_testbatch_2 = model.layers[1](test_x_batch)                                       # from case1
cells_h_for_testbatch_3 = model.predict(test_x_batch)                                                                  # from case1
cell2_h_for_testbatch_3 = cells_h_for_testbatch_3[:, -1, :]                                                            # from case1

cells_h_for_testbatch, cell2_h_for_testbatch = layer(test_x_batch, initial_state=[cell2_h_after_batch2])               # from case2
cells_h_for_testbatch - cells_h_for_testbatch_1
```
