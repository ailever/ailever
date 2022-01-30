## LSTM Cell
- https://en.wikipedia.org/wiki/Long_short-term_memory
![image](https://user-images.githubusercontent.com/56889151/151692355-f7cb33fc-4e81-4a62-a353-53031097a616.png)

```python
import tensorflow as tf
from tensorflow.keras import layers

# (batch, feature)
x = tf.random.normal(shape=(32, 8)) # x.shape # (32, 8) 
h = tf.random.normal(shape=(32, 4)) # h.shape # (32, 4) 
c = tf.random.normal(shape=(32, 4)) # c.shape # (32, 4)

layer = layers.LSTMCell(
    units=4, activation='tanh', recurrent_activation='sigmoid', 
    use_bias=True, unit_forget_bias=True,  
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal')
x_, (h_, c_) = layer(x, states=[h, c]) # x_ = h_

xW = tf.einsum('ij,jk->ik', x, layer.weights[0]) # xW.shape       # (32, 16)
hW = tf.einsum('ij,jk->ik', h, layer.weights[1]) # hW.shape       # (32, 16)
bilinear = xW + hW + layer.weights[2]            # bilinear.shape # (32, 16)

i = tf.sigmoid(bilinear[:, 0:4])   # i.shape # (32, 4)
f = tf.sigmoid(bilinear[:, 4:8])   # f.shape # (32, 4)
g = tf.tanh(bilinear[:, 8:12])     # g.shape # (32, 4)
o = tf.sigmoid(bilinear[:, 12:16]) # o.shape # (32, 4)
c = tf.einsum('ij,ij->ij', f, c) + tf.einsum('ij,ij->ij', i, g)
h = tf.einsum('ij,ij->ij', o, tf.tanh(c))

h - h_, c - c_
```
`layer weights information`
```python
layer.weights[0].shape # (8, 16)
layer.weights[1].shape # (4, 16)
layer.weights[2].shape # (, 16)
kernel_i = layer.weights[0][:, 0:4]      # kernel_i.shape    # (8, 4)
kernel_f = layer.weights[0][:, 4:8]      # kernel_f.shape    # (8, 4)
kernel_g = layer.weights[0][:, 8:12]     # kernel_g.shape    # (8, 4)
kernel_o = layer.weights[0][:, 12:16]    # kernel_o.shape    # (8, 4)
recurrent_i = layer.weights[1][:, 0:4]   # recurrent_i.shape # (4, 4)
recurrent_f = layer.weights[1][:, 4:8]   # recurrent_f.shape # (4, 4)
recurrent_g = layer.weights[1][:, 8:12]  # recurrent_g.shape # (4, 4)
recurrent_o = layer.weights[1][:, 12:16] # recurrent_o.shape # (4, 4)
bias_i = layer.weights[2][0:4]           # bias_i.shape      # (4, )
bias_f = layer.weights[2][4:8]           # bias_f.shape      # (4, )
bias_g = layer.weights[2][8:12]          # bias_g.shape      # (4, )
bias_o = layer.weights[2][12:16]         # bias_o.shape      # (4, )
```
