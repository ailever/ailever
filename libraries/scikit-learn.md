## [Machine Learning] | [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#) | [github](https://github.com/scikit-learn/scikit-learn)
### Classification
`[Classification]`

### Regression
`[Regression]: LinearRegression`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.boston(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = linear_model.LinearRegression()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
#regressor.coef_
#regressor.intercept_
regressor.predict(X[0:10])
```
`[Regression]: Ridge`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.boston(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = linear_model.Ridge(alpha=.5)
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
#regressor.coef_
#regressor.intercept_
regressor.predict(X[0:10])
```
`[Classification]: Lasso`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.boston(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = linear_model.Lasso(alpha=0.1)
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
#regressor.coef_
#regressor.intercept_
regressor.predict(X[0:10])
```

### Clustering

`[Clustering] kmeans`  
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(20)
X = np.random.normal(size=(1000, 2)) # 1000 pts in 2

# training
model = KMeans(n_clusters=3)
model.fit(X)
y_pred = model.predict(X) # labels for each of the points
centers = model.cluster_centers_ # locations of the clusters
labels = model.labels_

# visualization
center1 = np.where(y_pred == 0)
center2 = np.where(y_pred == 1)
center3 = np.where(y_pred == 2)

fig, axes = plt.subplots(1,2, figsize=(15,10))
axes[0].scatter(X[:,0], X[:,1])
axes[0].grid()
axes[1].scatter(X[:,0][center1], X[:,1][center1])
axes[1].scatter(X[:,0][center2], X[:,1][center2])
axes[1].scatter(X[:,0][center3], X[:,1][center3])
axes[1].grid()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/99382501-fcf38c80-290f-11eb-9672-bb8d2eaacd0e.png)


### Dimensionality Reduction
`[Dimensionality Reduction] `

### Model Selection
`[Model Selection] `

### Preprocessing
`[Preprocessing] `
