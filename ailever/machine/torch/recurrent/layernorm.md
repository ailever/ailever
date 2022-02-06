```python
import numpy as np
import torch
import torch.nn as nn

# [Batch, Sequence, Dimension]
x = 10*np.arange(10).reshape(-1, 5, 2)         # x.shape: (1,5,2)
embedding = torch.tensor(x).type(torch.float)

layer_norm = nn.LayerNorm(2)
norm1 = layer_norm(embedding)
norm2 = (embedding - embedding.mean(axis=-1).T) / torch.sqrt(embedding.var(axis=-1, unbiased=False).T + layer_norm.eps)
norm2 = torch.einsum('ijk,k->jk', [norm2, layer_norm.weight]) + layer_norm.bias
norm1, norm2
```
