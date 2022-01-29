
## Variance Inflation Factor

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

col0 = np.ones(X.shape[0])
col1 = np.linspace(-3, 3, 100)
col2 = np.linspace(-5, 5, 100)
col3 = np.sin(np.linspace(-3, 3, 100))
col4 = np.cos(np.linspace(-3, 3, 100))

X = np.c_[col0, col1, col2, col3, col4]
[variance_inflation_factor(X, i) for i in range(X.shape[1])]
```
