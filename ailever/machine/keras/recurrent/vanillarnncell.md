## Vanilla RNN
- https://en.wikipedia.org/wiki/Recurrent_neural_network
![image](https://user-images.githubusercontent.com/56889151/151692305-7a58363c-b9a2-483b-a814-2f35a90780ea.png)

```python
import tensorflow as tf
from tensorflow.keras import layers

# (batch, feature)
x = tf.random.normal(shape=(32, 8)) # x.shape # (32, 8) 
h = tf.random.normal(shape=(32, 4)) # h.shape # (32, 4) 

layer = layers.SimpleRNNCell(
    units=4, activation='tanh',  
    use_bias=True,   
    dropout=0, recurrent_dropout=0,
    kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
    kernel_constraint=None, recurrent_constraint=None, bias_constraint=None,
    bias_initializer='zeros', kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal')
x_, (h_, ) = layer(x, states=[h]) # x_ = h_

# layer.weights[0].shape # (8, 4)
# layer.weights[1].shape # (4, 4)
# layer.weights[2].shape # (, 4)
xW = tf.einsum('ij,jk->ik', x, layer.weights[0]) # xW.shape       # (32, 4)
hW = tf.einsum('ij,jk->ik', h, layer.weights[1]) # hW.shape       # (32, 4)
bilinear = xW + hW + layer.weights[2]            # bilinear.shape # (32, 4)
h = tf.tanh(bilinear)                            # h.shape        # (32, 4)

h - h_
```
