## Bidirectional RNN(Recurrent Neural Network)
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional

### Bidirectional SimpleRNN
```python
import tensorflow as tf
from tensorflow.keras import layers

# [BatchFirst](batch, sequence, feature)
x = tf.random.normal(shape=(32, 2, 8))                                                                          # x.shape                # (32, 2, 8) 
cell0_h = tf.random.normal(shape=(32, 4))                                                                       # cell0_h.shape          # (32, 4) 

forward_rnn = layers.SimpleRNN(4, stateful=False, time_major=False, unroll=False, return_sequences=True, return_state=True, go_backwards=False)
backward_rnn = layers.SimpleRNN(4, stateful=False, time_major=False, unroll=False, return_sequences=True, return_state=True, go_backwards=True)
bidirectional_rnn = layers.Bidirectional(forward_rnn, backward_layer=backward_rnn, merge_mode='concat')
whole_h_, forward_h_, backward_h_ = bidirectional_rnn(x, initial_state=[cell0_h, cell0_h])
forward_hs, forward_h = bidirectional_rnn.forward_layer(x, initial_state=[cell0_h])
backward_hs, backward_h = bidirectional_rnn.backward_layer(x, initial_state=[cell0_h])

#whole_h_[:, :, 0:4] - forward_hs
#whole_h_[:, :, 4:8] - tf.reverse(backward_hs, axis=[1]) # (time_major=False --> axis=1) / (time_major=True --> axis=0)
forward_h_ - forward_h
backward_h_ - backward_h
```

### Bidirectional LSTM
```python
import tensorflow as tf
from tensorflow.keras import layers

# [BatchFirst](batch, sequence, feature)
x = tf.random.normal(shape=(32, 2, 8))                                                                          # x.shape                # (32, 2, 8) 
cell0_h = tf.random.normal(shape=(32, 4))                                                                       # cell0_h.shape          # (32, 4) 
cell0_c = tf.random.normal(shape=(32, 4))                                                                       # cell0_c.shape          # (32, 4) 

forward_rnn = layers.LSTM(4, stateful=False, time_major=False, unroll=False, return_sequences=True, return_state=True, go_backwards=False)
backward_rnn = layers.LSTM(4, stateful=False, time_major=False, unroll=False, return_sequences=True, return_state=True, go_backwards=True)
bidirectional_rnn = layers.Bidirectional(forward_rnn, backward_layer=backward_rnn, merge_mode='concat')
whole_h_, forward_h_, forward_c_, backward_h_, backward_c_ = bidirectional_rnn(x, initial_state=[cell0_h, cell0_c, cell0_h, cell0_c])
forward_hs, forward_h, forward_c = bidirectional_rnn.forward_layer(x, initial_state=[cell0_h, cell0_c])
backward_hs, backward_h, backward_c = bidirectional_rnn.backward_layer(x, initial_state=[cell0_h, cell0_c])

#whole_h_[:, :, 0:4] - forward_hs
#whole_h_[:, :, 4:8] - tf.reverse(backward_hs, axis=[1]) # (time_major=False --> axis=1) / (time_major=True --> axis=0)
forward_h_ - forward_h
backward_h_ - backward_h
```

### Bidirectional GRU
```python
import tensorflow as tf
from tensorflow.keras import layers

# [BatchFirst](batch, sequence, feature)
x = tf.random.normal(shape=(32, 2, 8))                                                                          # x.shape                # (32, 2, 8) 
cell0_h = tf.random.normal(shape=(32, 4))                                                                       # cell0_h.shape          # (32, 4) 

forward_rnn = layers.GRU(4, stateful=False, time_major=False, unroll=False, return_sequences=True, return_state=True, go_backwards=False)
backward_rnn = layers.GRU(4, stateful=False, time_major=False, unroll=False, return_sequences=True, return_state=True, go_backwards=True)
bidirectional_rnn = layers.Bidirectional(forward_rnn, backward_layer=backward_rnn, merge_mode='concat')
whole_h_, forward_h_, backward_h_ = bidirectional_rnn(x, initial_state=[cell0_h, cell0_h])
forward_hs, forward_h = bidirectional_rnn.forward_layer(x, initial_state=[cell0_h])
backward_hs, backward_h = bidirectional_rnn.backward_layer(x, initial_state=[cell0_h])

#whole_h_[:, :, 0:4] - forward_hs
#whole_h_[:, :, 4:8] - tf.reverse(backward_hs, axis=[1]) # (time_major=False --> axis=1) / (time_major=True --> axis=0)
forward_h_ - forward_h
backward_h_ - backward_h
```
