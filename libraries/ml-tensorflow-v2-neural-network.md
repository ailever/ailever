
## Recurrent Neural Network
### LSTM
- https://en.wikipedia.org/wiki/Long_short-term_memory
![image](https://user-images.githubusercontent.com/56889151/151684845-57d8fb6f-7a4e-436e-882a-9f24f9826026.png)

```python
import tensorflow as tf
from tensorflow.keras import models, layers

X = tf.random.normal(shape=[32, 10, 8])
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
tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
```
![image](https://user-images.githubusercontent.com/56889151/151684225-0eb620d2-59bf-40b1-9e97-722125b86380.png)

```python
import tensorflow as tf
from tensorflow.keras import layers

X = layers.Input(shape=(10, 8))
layer = layers.LSTM(4, return_sequences=True, return_state=True)
layer(X)

layer.weights
layer.recurrent_activation
```

