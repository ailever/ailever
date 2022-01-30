
## GRU(Gated Recurrent Unit) Cell
- https://en.wikipedia.org/wiki/Gated_recurrent_unit
- https://stackoverflow.com/questions/57318930/calculating-the-number-of-parameters-of-a-gru-layer-keras
![image](https://user-images.githubusercontent.com/56889151/151692414-f8cf6f5d-1313-4274-8315-ec04f877057f.png)

`reset_after=True (tf version2 default)`
```python
import tensorflow as tf
from tensorflow.keras import layers

# (batch, feature)
x = tf.random.normal(shape=(32, 8)) # x.shape # (32, 8) 
h = tf.random.normal(shape=(32, 4)) # h.shape # (32, 4) 

cell = layers.GRUCell(
    units=4, activation='tanh', recurrent_activation='sigmoid', 
    use_bias=True,
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', reset_after=True)
x_, (h_, ) = cell(x, states=[h]) # x_ = h_

#cell.weights[0].shape # (8, 12)
#cell.weights[1].shape # (4, 12)
#cell.weights[2].shape # (2, 12)
W_z = cell.weights[0][:, 0:4]       # W_z.shape  # (8,4)
W_r = cell.weights[0][:, 4:8]       # W_r.shape  # (8,4)
W_c = cell.weights[0][:, 8:12]      # W_c.shape  # (8,4)
U_z = cell.weights[1][:, 0:4]       # U_z.shape  # (4,4)
U_r = cell.weights[1][:, 4:8]       # U_r.shape  # (4,4)
U_c = cell.weights[1][:, 8:12]      # U_c.shape  # (4,4)
b0_z = cell.weights[2][0, 0:4]      # b0_z.shape # (, 4)
b0_r = cell.weights[2][0, 4:8]      # b0_r.shape # (, 4)
b0_c = cell.weights[2][0, 8:12]     # b0_c.shape # (, 4)
b1_z = cell.weights[2][1, 0:4]      # b1_z.shape # (, 4)
b1_r = cell.weights[2][1, 4:8]      # b1_r.shape # (, 4)
b1_c = cell.weights[2][1, 8:12]     # b1_c.shape # (, 4)

xW_z = tf.einsum('ij,jk->ik', x, W_z)                                               # xW_z.shape # (32, 4)
hU_z = tf.einsum('ij,jk->ik', h, U_z)                                               # hU_z.shape # (32, 4)
xW_r = tf.einsum('ij,jk->ik', x, W_r)                                               # xW_r.shape # (32, 4)
hU_r = tf.einsum('ij,jk->ik', h, U_r)                                               # hU_r.shape # (32, 4)
z = tf.sigmoid((xW_z + b0_z) + (hU_z + b1_z))                                       # z.shape    # (32, 4)
r = tf.sigmoid((xW_r + b0_r) + (hU_r + b1_r))                                       # r.shape    # (32, 4)
xW_c = tf.einsum('ij,jk->ik', x, W_c)                                               # xW_c.shape # (32, 4)
transformed_hU_c = tf.einsum('ij,ij->ij', r, tf.einsum('ij,jk->ik', h, U_c) + b1_c) # transformed_hU_c.shape # (32, 4)
c = tf.tanh((xW_c + b0_c) + transformed_hU_c)                                       # c.shape    # (32, 4)

h = tf.einsum('ij,ij->ij', (1-z), c) + tf.einsum('ij,ij->ij', z, h)
h - h_
```

`reset_after=False (tf version1 default)`
```python
import tensorflow as tf
from tensorflow.keras import layers

# (batch, feature)
x = tf.random.normal(shape=(32, 8)) # x.shape # (32, 8) 
h = tf.random.normal(shape=(32, 4)) # h.shape # (32, 4) 

layer = layers.GRUCell(
    units=4, activation='tanh', recurrent_activation='sigmoid', 
    use_bias=True,
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', reset_after=False)
x_, (h_, ) = layer(x, states=[h]) # x_ = h_

#layer.weights[0].shape # (8, 12)
#layer.weights[1].shape # (4, 12)
#layer.weights[2].shape # (, 12)
W_z = layer.weights[0][:, 0:4]   # W_z.shape # (8,4)
W_r = layer.weights[0][:, 4:8]   # W_r.shape # (8,4)
W_c = layer.weights[0][:, 8:12]  # W_c.shape # (8,4)
U_z = layer.weights[1][:, 0:4]   # U_z.shape # (4,4)
U_r = layer.weights[1][:, 4:8]   # U_r.shape # (4,4)
U_c = layer.weights[1][:, 8:12]  # U_c.shape # (4,4)
b_z = layer.weights[2][0:4]      # b_z.shape # (, 4)
b_r = layer.weights[2][4:8]      # b_r.shape # (, 4)
b_c = layer.weights[2][8:12]     # b_c.shape # (, 4)

xW_z = tf.einsum('ij,jk->ik', x, W_z)                                         # xW_z.shape              # (32, 4)
hU_z = tf.einsum('ij,jk->ik', h, U_z)                                         # hU_z.shape              # (32, 4)
xW_r = tf.einsum('ij,jk->ik', x, W_r)                                         # xW_r.shape              # (32, 4)
hU_r = tf.einsum('ij,jk->ik', h, U_r)                                         # hU_r.shape              # (32, 4)
z = tf.sigmoid(xW_z + hU_z + b_z)                                             # z.shape                 # (32, 4)
r = tf.sigmoid(xW_r + hU_r + b_r)                                             # r.shape                 # (32, 4)
xW_c = tf.einsum('ij,jk->ik', x, W_c)                                         # xW_c.shape              # (32, 4)
trainsformed_hU_c = tf.einsum('ij,jk->ik', tf.einsum('ij,ij->ij', h, r), U_c) # trainsformed_hU_c.shape # (32, 4)
c = tf.tanh(xW_c + trainsformed_hU_c + b_c)                                   # c.shape                 # (32, 4)

h = tf.einsum('ij,ij->ij', (1-z), c) + tf.einsum('ij,ij->ij', z, h)
h - h_
```

