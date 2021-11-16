## [Machine Learning] | [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#) | [github](https://github.com/scikit-learn/scikit-learn)
### Classification
#### Classification: Ensemble
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
classifier = ensemble.GradientBoostingClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
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
classifier = ensemble.RandomForestClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
#### Classification: naive_bayes
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
classifier = naive_bayes.GaussianNB()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
#### Classification: tree
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
classifier = tree.DecisionTreeClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
#### Classification: neighbors
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
classifier = neighbors.KNeighborsClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
#### Classification: discriminant_analysis
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
classifier = discriminant_analysis.LinearDiscriminantAnalysis()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
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
classifier = discriminant_analysis.QuadraticDiscriminantAnalysis()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
#### Classification: linear_model
`[Classification]: LogisticRegression`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = linear_model.LogisticRegression(penalty='l2', max_iter=500)
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
#### Classification: svm
`[Classification]: LinearSVC`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import svm

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = svm.LinearSVC(max_iter=10000)
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classification]: SVC`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import svm

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = svm.SVC(kernel='poly', max_iter=10000) # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
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
#### Regression: linear_model
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
#### Regression: svm
`[Regression]: LinearSVR`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import svm

# [STEP1]: data
dataset = SKAPI.boston(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = svm.LinearSVR(max_iter=1000)
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: SVR`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import svm

# [STEP1]: data
dataset = SKAPI.boston(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = svm.SVR(kernel='poly', max_iter=1000) # kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: `
```python
```
`[Regression]: `
```python
```
`[Regression]: `
```python
```
`[Regression]: `
```python
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


### Dimensionality Reduction
`[Dimensionality Reduction]: Isomap`
```python
import numpy as np
from sklearn.manifold import Isomap

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = Isomap(n_components=2)
X_embeded = model.fit_transform(X); print(X_embeded)
```
`[Dimensionality Reduction]: LocallyLinearEmbedding-hessian`
```python
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = LocallyLinearEmbedding(n_neighbors=6, n_components=2, method='hessian')
X_embeded = model.fit_transform(X[:100]); print(X_embeded)
```
`[Dimensionality Reduction]: LocallyLinearEmbedding-ltsa`
```python
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = LocallyLinearEmbedding(n_neighbors=5, n_components=2, method='ltsa')
X_embeded = model.fit_transform(X[:100]); print(X_embeded)
```
`[Dimensionality Reduction]: LocallyLinearEmbedding-modified`
```python
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = LocallyLinearEmbedding(n_neighbors=5, n_components=2, method='modified')
X_embeded = model.fit_transform(X[:100]); print(X_embeded)
```
`[Dimensionality Reduction]: LocallyLinearEmbedding-standard`
```python
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = LocallyLinearEmbedding(n_neighbors=5, n_components=2, method='standard')
X_embeded = model.fit_transform(X[:100]); print(X_embeded)
```
`[Dimensionality Reduction]: MDS`
```python
import numpy as np
from sklearn.manifold import MDS

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = MDS(n_components=2)
X_embeded = model.fit_transform(X[:100]); print(X_embeded)
```
`[Dimensionality Reduction]: SpectralEmbedding`
```python
import numpy as np
from sklearn.manifold import SpectralEmbedding

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = SpectralEmbedding(n_components=2)
X_embeded = model.fit_transform(X); print(X_embeded)
```
`[Dimensionality Reduction]: TSNE`
```python
import numpy as np
from sklearn.manifold import TSNE

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = TSNE(n_components=2)
X_embeded = model.fit_transform(X); print(X_embeded)
```

### Model Selection
`[Model Selection] `

### Preprocessing
`[Preprocessing] `
