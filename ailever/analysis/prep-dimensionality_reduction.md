
```python
import numpy as np
from sklearn.manifold import Isomap
from ailever.dataset import SKAPI

dataset = SKAPI.digits(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, 'target'].values.ravel()

reduction = Isomap()
X_new = reduction.fit_transform(X)
X_new
```


```python
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
from ailever.dataset import SKAPI

dataset = SKAPI.digits(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, 'target'].values.ravel()

reduction = LocallyLinearEmbedding()
X_new = reduction.fit_transform(X)
X_new
```


```python
import numpy as np
from sklearn.manifold import MDS
from ailever.dataset import SKAPI

dataset = SKAPI.digits(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, 'target'].values.ravel()

reduction = MDS()
X_new = reduction.fit_transform(X)
X_new
```


```python
import numpy as np
from sklearn.manifold import SpectralEmbedding
from ailever.dataset import SKAPI

dataset = SKAPI.digits(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, 'target'].values.ravel()

reduction = SpectralEmbedding()
X_new = reduction.fit_transform(X)
X_new
```


```python
import numpy as np
from sklearn.manifold import TSNE
from ailever.dataset import SKAPI

dataset = SKAPI.digits(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, 'target'].values.ravel()

reduction = TSNE()
X_new = reduction.fit_transform(X)
X_new
```
