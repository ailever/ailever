
## Regression Factor Analysis
### Analytic Inference
`statsmodels`
```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

white_noise = np.random.normal(size=100)
time_series = np.zeros_like(white_noise)
for t, noise in enumerate(white_noise):
    time_series[t] = time_series[t-1] + noise

Y = time_series
X = np.arange(len(Y))
X_ = sm.add_constant(X, has_constant='add')
model = sm.OLS(Y,X_).fit()
plt.plot(time_series)
plt.plot(model.predict(X_))
plt.plot(model.params[1]*X+ model.params[0], ls='--')
```
`numpy`
```python
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt

data = np.array([[0.05, 0.12],
                 [0.18, 0.22],
                 [0.31, 0.35],
                 [0.42, 0.38],
                 [0.5, 0.49]])

x, y = data[:,0], data[:,1]
bias = np.ones_like(x)
X = np.c_[bias, x]

b = linalg.inv(X.T@X) @ X.T @ y
yhat = X@b

plt.scatter(x, y)
plt.plot(x, yhat, color='red')
plt.show()
```



