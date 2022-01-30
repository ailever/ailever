```python
import tensorflow as tf
from tensorflow.keras import models, layers, utils

X = tf.random.normal(shape=(32, 2, 2))
layer = layers.LSTM(4, return_sequences=False, return_state=False)
y = layer(X) 
y.shape #: (32, 4), last_seq_output

layer = layers.LSTM(4, return_sequences=True, return_state=False)
y = layer(X)
y.shape #: (32,10, 4), whole_seq_output

layer = layers.LSTM(4, return_sequences=True, return_state=True)
y = layer(X)
y[0].shape #: (32, 10, 4), whole_seq_output
y[1].shape #: (32, 4), final_memory_state 
y[2].shape #: (32, 4), final_carry_state

layer = layers.LSTM(4, return_sequences=False, return_state=True)
y = layer(X)
y[0].shape #: (32, 4), last_seq_output
y[1].shape #: (32, 4), final_memory_state 
y[2].shape #: (32, 4), final_carry_state

# visualization
X = layers.Input(shape=(10, 8))
layer = layers.LSTM(4, return_sequences=True, return_state=True)
model = models.Model(X, layer(X))
utils.plot_model(model, "model.png", show_shapes=True)
```
