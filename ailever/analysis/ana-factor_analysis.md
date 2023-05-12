## Factor Analysis
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from sklearn.preprocessing import normalize

# Figure Setting
fig = plt.figure(figsize=(30,5))
ax1 = plt.subplot2grid((1,4), (0,0), fig=fig)
ax2 = plt.subplot2grid((1,4), (0,1), fig=fig, colspan=3)

major_ticks = np.arange(-10, 10, 2)
minor_ticks = np.arange(-10, 10, 0.5)

ax1.set_xticks(major_ticks, minor=False)
ax1.set_xticks(minor_ticks, minor=True)
ax1.set_yticks(major_ticks, minor=False)
ax1.set_yticks(minor_ticks, minor=True)

#ax.grid(which='both', alpha=0.2)
ax1.grid(which='minor', alpha=0.2)
ax1.grid(which='major', alpha=0.5)

# Data & Factor Analysis
angle = 0 #np.pi/6
noise = 0.000001
df = pd.DataFrame(
    data=[
        [ np.cos(angle) + np.random.normal(0, noise), -np.sin(angle) + np.random.normal(0, noise), noise*np.random.normal(0, noise), noise*np.random.normal(0, noise)],
        [ np.sin(angle) + np.random.normal(0, noise),  np.cos(angle) + np.random.normal(0, noise), noise*np.random.normal(0, noise), noise*np.random.normal(0, noise)],
        [ np.cos(angle) + np.random.normal(0, noise), -np.sin(angle) + np.random.normal(0, noise), noise*np.random.normal(0, noise), noise*np.random.normal(0, noise)],
        [ np.sin(angle) + np.random.normal(0, noise),  np.cos(angle) + np.random.normal(0, noise), noise*np.random.normal(0, noise), noise*np.random.normal(0, noise)],
        [ np.cos(angle) + np.random.normal(0, noise), -np.sin(angle) + np.random.normal(0, noise), noise*np.random.normal(0, noise), noise*np.random.normal(0, noise)],
        [ np.sin(angle) + np.random.normal(0, noise),  np.cos(angle) + np.random.normal(0, noise), noise*np.random.normal(0, noise), noise*np.random.normal(0, noise)],
        [ np.cos(angle) + np.random.normal(0, noise), -np.sin(angle) + np.random.normal(0, noise), noise*np.random.normal(0, noise), noise*np.random.normal(0, noise)],
        [ np.sin(angle) + np.random.normal(0, noise),  np.cos(angle) + np.random.normal(0, noise), noise*np.random.normal(0, noise), noise*np.random.normal(0, noise)],
        [ np.random.normal(0, noise)                ,  np.random.normal(0, noise)                , noise*np.random.normal(0, noise), noise*np.random.normal(0, noise)],              
        [ np.random.normal(0, noise)                ,  np.random.normal(0, noise)                , noise*np.random.normal(0, noise), noise*np.random.normal(0, noise)]
    ]
)
df.values[:, :] = normalize(df, axis=0)
fa = FactorAnalyzer(n_factors=2, method="ml", rotation="promax")
fa.fit(df)

# fa.loadings_
# fa.get_communalities()
# fa.get_eigenvalues()

# Visualization
ax1.scatter([0], [0], c='black')
ax1.scatter(df.values[0], df.values[1], marker='^')
sns.heatmap(fa.loadings_, cmap="Blues", annot=True, fmt='.2f', ax=ax2)
```


```python
# Caution: The module FAMD not consider any rotations in feature space.

import numpy as np
import pandas as pd
from prince import FAMD
# Factor Analysis of Mixed Data (FAMD)
# :FAMD does the analysis with a combination of PCA and MCA techniques. MCA stands for Multiple Correspondence Analysis which is suitable for multiple categorical factors specifically.
# https://towardsdatascience.com/factor-analysis-of-mixed-data-5ad5ce98663c

df = pd.DataFrame(data=np.c_[np.random.randint(10, size=(300,1)), np.random.normal(5, size=(300,2))], columns=['X0', 'X1', 'X2'])
df['X0'] = df['X0'].astype('category')

famd = FAMD(n_components=2, n_iter=10, random_state=101)
famd.fit(df)
famd.transform(df) # famd.row_coordinates(df)
famd.column_coordinates_
```



---


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
![image](https://user-images.githubusercontent.com/56889151/150993922-924aadff-6369-4c50-9beb-37c00ae0770e.png)
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



