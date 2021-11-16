## [Machine Learning] | [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#) | [github](https://github.com/scikit-learn/scikit-learn)
### Classification
`[Classification]: GradientBoostingClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = ensemble.GradientBoostingClassifier()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'classifier.joblib')
regressor = joblib.load('classifier.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Classfication]: RandomForestClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = ensemble.RandomForestClassifier()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'classifier.joblib')
regressor = joblib.load('classifier.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Classification]: GaussianNB`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import naive_bayes

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = naive_bayes.GaussianNB()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'classifier.joblib')
regressor = joblib.load('classifier.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Classification]: DecisionTreeClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import tree

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = tree.DecisionTreeClassifier()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'classifier.joblib')
regressor = joblib.load('classifier.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Classification]: KNeighborsClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import neighbors


# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = neighbors.KNeighborsClassifier()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'classifier.joblib')
regressor = joblib.load('classifier.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Classification]: LinearDiscriminantAnalysis`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import discriminant_analysis


# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = discriminant_analysis.LinearDiscriminantAnalysis()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'classifier.joblib')
regressor = joblib.load('classifier.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Classification]: QuadraticDiscriminantAnalysis`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import discriminant_analysis


# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = discriminant_analysis.QuadraticDiscriminantAnalysis()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'classifier.joblib')
regressor = joblib.load('classifier.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Classification]: `
```python
```
`[Classification]: `
```python
```
`[Classification]: `
```python
```
`[Classification]: `
```python
```
`[Classification]: `
```python
```
`[Classification]: `
```python
```


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
`[Regression]: Lasso`
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
`[Regression]: LassoLars`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.boston(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = linear_model.LassoLars(alpha=.1, normalize=False)
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
#regressor.coef_
#regressor.intercept_
regressor.predict(X[0:10])
```
`[Regression]: ElasticNet`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.boston(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = linear_model.ElasticNet()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
#regressor.coef_
#regressor.intercept_
regressor.predict(X[0:10])
```
`[Regression]: BayesianRidge`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.boston(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = linear_model.BayesianRidge()
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
