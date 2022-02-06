```python
import numpy as np
import torch
import torch.nn as nn

x = 10*np.arange(10).reshape(5, 2)
embedding = torch.tensor(x).reshape(-1, *x.shape).type(torch.float)

layer_norm = nn.LayerNorm(2)
norm1 = layer_norm(embedding)
norm2 = (embedding - embedding.mean(axis=-1).T) / embedding.std(axis=-1, unbiased=False).T
norm1, norm2
```
