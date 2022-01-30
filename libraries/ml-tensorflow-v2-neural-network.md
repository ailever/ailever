
## Recurrent Neural Network
### LSTM
```python
import tensorflow as tf
from tensorflow.keras import layers

X = tf.random.normal(shape=[32, 10, 8])

model = layers.LSTM(4, return_sequences=False, return_state=False)
y = model(X) 
y.shape #: (32, 4), , last_seq_output

model = layers.LSTM(4, return_sequences=True, return_state=False)
y = model(X)
y.shape #: (32,10, 4), whole_seq_output

model = layers.LSTM(4, return_sequences=True, return_state=True)
y = model(X)
y[0].shape #: (32, 10, 4), whole_seq_output
y[1].shape #: (32, 4), final_memory_state 
y[2].shape #: (32, 4), final_carry_state

model = layers.LSTM(4, return_sequences=False, return_state=True)
y = model(X)
y[0].shape #: (32, 4), last_seq_output
y[1].shape #: (32, 4), final_memory_state 
y[2].shape #: (32, 4), final_carry_state
```
