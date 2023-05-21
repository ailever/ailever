## [Data Analysis] | [pasty](https://patsy.readthedocs.io/en/latest/index.html) | [github](https://github.com/pydata/patsy/blob/master/doc/overview.rst) 

## Pasty-Basic



```python
import numpy as np
import pandas as pd
from patsy import dmatrix, dmatrices

def function(x):
    return 2*x

np.random.seed(0)
x1 = np.random.rand(5) + 10
x2 = np.random.rand(5) * 10

dmatrix("x1")
dmatrix("x1 - 1")
dmatrix("x1 + 0")
dmatrix("x1 + x2")
dmatrix("x1 + x2 - 1")
dmatrix("function(x1)")

df = pd.DataFrame(data=np.c_[np.random.normal(0, size=5), np.random.normal(1, size=5), np.random.poisson(10, size=5)] ,columns=['N1', 'N2', 'N3'])
df['C1'] = np.array(['P', 'Q', 'R', 'S', 'P'], dtype=str)

dmatrix("N1 + N2 - 1", data=df)
dmatrix("N1 + np.log(np.abs(N2))", data=df)
dmatrix("N1 + N2 + np.log(np.abs(N2)) + I(np.log(np.abs(N2)))", data=df)

dmatrix("N1 + function(N1)", data=df)
dmatrix("N1 + center(N1) + standardize(N1)", data=df)
dmatrix("N2", data=df)
dmatrix("N3", data=df)
dmatrix("C1", data=df)
dmatrix("C1 + 0", data=df)
dmatrix("C(N3)", data=df)
dmatrix("C(N3) + 0", data=df)

np.asarray(dmatrix("C1", data=df))
```
