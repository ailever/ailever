
## Variance Inflation Factor

```python
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

size = 100
col0 = np.ones(size)
col1 = np.linspace(-3, 3, size)
col2 = np.linspace(-5, 5, size)
col3 = np.sin(np.linspace(-3, 3, size))
col4 = np.cos(np.linspace(-3, 3, size))
col5 = np.exp(np.linspace(-3, 3, size))
col6 = np.log(np.linspace(0.1, 3, size))

X = np.c_[col0, col1, col2, col3, col4, col5, col6]
vifs = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

print(vifs)
plt.figure(figsize=(25,5))
plt.barh(range(X.shape[1]), vifs)
```
