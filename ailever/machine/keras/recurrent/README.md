## Recurrent Neural Network
- https://www.tensorflow.org/api_docs/python/tf/keras/layers

`RNN Cell`
```python
import tensorflow as tf
from tensorflow.keras import layers

x = tf.random.normal(shape=(32, 8))                         # x.shape    # (32, 8) 
h = tf.random.normal(shape=(32, 4))                         # h.shape    # (32, 4) 
c = tf.random.normal(shape=(32, 4))                         # h.shape    # (32, 4) 
hs_, (h_, ) = layers.SimpleRNNCell(units=4)(x, states=[h])  # hs_.shape  # (32, 4), hs_ = h_
                                                            # h_.shape   # (32, 4), hs_ = h_ 
hs_, (h_, c_) = layers.LSTMCell(units=4)(x, states=[h, c])  # hs_.shape  # (32, 4), hs_ = h_
                                                            # h_.shape   # (32, 4), hs_ = h_ 
                                                            # c_.shape   # (32, 4) 
hs_, (h_, ) = layers.GRUCell(units=4)(x, states=[h])        # hs_.shape  # (32, 4), hs_ = h_
                                                            # h_.shape   # (32, 4), hs_ = h_
```

`RNN Layer: BatchFirst`
```python
import tensorflow as tf
from tensorflow.keras import layers

x = tf.random.normal(shape=(32, 2, 8))                                                                                   # x.shape    # (32, 2, 8) 
h = tf.random.normal(shape=(32, 4))                                                                                      # h.shape    # (32, 4) 
c = tf.random.normal(shape=(32, 4))                                                                                      # h.shape    # (32, 4) 
hs_, h_ = layers.SimpleRNN(units=4, time_major=False, return_sequences=True, return_state=True)(x, initial_state=[h])    # hs_.shape  # (32, 2, 4), hs_[-1] = h_
                                                                                                                         # h_.shape   # (32, 4),    hs_[-1] = h_ 
hs_, h_ = layers.SimpleRNN(units=4, time_major=False, return_sequences=False, return_state=True)(x, initial_state=[h])   # hs_.shape  # (32, 4),    hs_ = h_
                                                                                                                         # h_.shape   # (32, 4),    hs_ = h_ 
hs_ = layers.SimpleRNN(units=4, time_major=False, return_sequences=False, return_state=False)(x, initial_state=[h])      # hs_.shape  # (32, 4),    hs_
hs_, h_, c_ = layers.LSTM(units=4, time_major=False, return_sequences=True, return_state=True)(x, initial_state=[h, c])  # hs_.shape  # (32, 2, 4), hs_[-1] = h_
                                                                                                                         # h_.shape   # (32, 4),    hs_[-1] = h_ 
                                                                                                                         # c_.shape   # (32, 4)
hs_, h_, c_ = layers.LSTM(units=4, time_major=False, return_sequences=False, return_state=True)(x, initial_state=[h, c]) # hs_.shape  # (32, 4),    hs_ = h_
                                                                                                                         # h_.shape   # (32, 4),    hs_ = h_
                                                                                                                         # c_.shape   # (32, 4)    
hs_ = layers.LSTM(units=4, time_major=False, return_sequences=False, return_state=False)(x, initial_state=[h, c])        # hs_.shape  # (32, 4)
hs_, h_ = layers.GRU(units=4, time_major=False, return_sequences=True, return_state=True)(x, initial_state=[h])          # hs_.shape  # (32, 2, 4), hs_[-1] = h_
                                                                                                                         # h_.shape   # (32, 4),    hs_[-1] = h_ 
hs_, h_ = layers.GRU(units=4, time_major=False, return_sequences=False, return_state=True)(x, initial_state=[h])         # hs_.shape  # (32, 4),    hs_ = h_
                                                                                                                         # h_.shape   # (32, 4),    hs_ = h_ 
hs_ = layers.GRU(units=4, time_major=False, return_sequences=False, return_state=False)(x, initial_state=[h])            # hs_.shape  # (32, 4),    hs_
```

`RNN Layer: TimeMajor`
```python
import tensorflow as tf
from tensorflow.keras import layers

x = tf.random.normal(shape=(2, 32, 8))                                                                                   # x.shape    # (2, 32, 8) 
h = tf.random.normal(shape=(32, 4))                                                                                      # h.shape    # (32, 4) 
c = tf.random.normal(shape=(32, 4))                                                                                      # h.shape    # (32, 4) 
hs_, h_ = layers.SimpleRNN(units=4, time_major=True, return_sequences=True, return_state=True)(x, initial_state=[h])     # hs_.shape  # (2, 32, 4), hs_[-1] = h_
                                                                                                                         # h_.shape   # (32, 4),    hs_[-1] = h_ 
hs_, h_ = layers.SimpleRNN(units=4, time_major=True, return_sequences=False, return_state=True)(x, initial_state=[h])    # hs_.shape  # (32, 4),    hs_ = h_
                                                                                                                         # h_.shape   # (32, 4),    hs_ = h_ 
hs_ = layers.SimpleRNN(units=4, time_major=True, return_sequences=False, return_state=False)(x, initial_state=[h])       # hs_.shape  # (32, 4),    hs_
hs_, h_, c_ = layers.LSTM(units=4, time_major=True, return_sequences=True, return_state=True)(x, initial_state=[h, c])   # hs_.shape  # (2, 32, 4), hs_[-1] = h_
                                                                                                                         # h_.shape   # (32, 4),    hs_[-1] = h_ 
                                                                                                                         # c_.shape   # (32, 4)
hs_, h_, c_ = layers.LSTM(units=4, time_major=True, return_sequences=False, return_state=True)(x, initial_state=[h, c])  # hs_.shape  # (32, 4),    hs_ = h_
                                                                                                                         # h_.shape   # (32, 4),    hs_ = h_
                                                                                                                         # c_.shape   # (32, 4)    
hs_ = layers.LSTM(units=4, time_major=True, return_sequences=False, return_state=False)(x, initial_state=[h, c])         # hs_.shape  # (32, 4)
hs_, h_ = layers.GRU(units=4, time_major=True, return_sequences=True, return_state=True)(x, initial_state=[h])           # hs_.shape  # (2, 32, 4), hs_[-1] = h_
                                                                                                                         # h_.shape   # (32, 4),    hs_[-1] = h_ 
hs_, h_ = layers.GRU(units=4, time_major=True, return_sequences=False, return_state=True)(x, initial_state=[h])          # hs_.shape  # (32, 4),    hs_ = h_
                                                                                                                         # h_.shape   # (32, 4),    hs_ = h_ 
hs_ = layers.GRU(units=4, time_major=True, return_sequences=False, return_state=False)(x, initial_state=[h])             # hs_.shape  # (32, 4),    hs_
```
