## Attention 
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention

```python
import numpy as np
import tensorflow as tf

q = tf.random.normal(shape=(32, 2, 8))                         # q.shape    # (32, 2, 8) 
k = tf.random.normal(shape=(32, 2, 8))                         # k.shape    # (32, 2, 8) 
v = tf.random.normal(shape=(32, 2, 8))                         # v.shape    # (32, 2, 8) 
attention_mask = tf.constant(np.tril(np.full(fill_value=1, shape=(32, 2))), dtype=tf.bool)

layer = tf.keras.layers.Attention(use_scale=False, causal=True, dropout=0)
sequence1 = layer([q,q,q], mask=[attention_mask, attention_mask])
sequence1 = layer([q,q,q])

attention_mask = tf.constant(np.tril(np.full(fill_value=1, shape=(32, 2, 2))), dtype=tf.float32)

scores = tf.einsum('ijk,ilk->ijl', q,q)#/tf.math.sqrt(tf.constant(q.shape[-1], dtype=tf.float32))
masked_scores = tf.einsum('ijk,ijk->ijk', scores, attention_mask)
distribution = tf.nn.softmax(masked_scores, axis=-1)

masked_distribution = tf.einsum('ijk,ijk->ijk', distribution, attention_mask)
sequence2 = tf.einsum('ijl,ilm->ijm', masked_distribution, q)
sequence1 - sequence2
```
