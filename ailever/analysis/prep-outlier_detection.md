## Isolation Forest
```python
import pandas as pd
from sklearn.ensemble import IsolationForest
from ailever.dataset import SKAPI

# load the dataset
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, 'target'].values.ravel()

print('- ORIGIN: ', X.shape, y.shape)
outlier_detector = IsolationForest(contamination=0.1)
anomaly = outlier_detector.fit_predict(X)

# select all rows that are not outliers
mask = anomaly != -1
X_new, y_new = X[mask, :], y[mask]
print('- NEW:', X_new.shape, y_new.shape)
```

## Minimum Covariance Determinant
```python
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from ailever.dataset import SKAPI

# load the dataset
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, 'target'].values.ravel()

print('- ORIGIN: ', X.shape, y.shape)
outlier_detector = EllipticEnvelope(contamination=0.01)

anomaly = outlier_detector.fit_predict(X)
# select all rows that are not outliers
mask = anomaly != -1
X_new, y_new = X[mask, :], y[mask]
print('- NEW:', X_new.shape, y_new.shape)
```

## Local Outlier Factor
```python
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from ailever.dataset import SKAPI

# load the dataset
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, 'target'].values.ravel()

print('- ORIGIN: ', X.shape, y.shape)
outlier_detector = LocalOutlierFactor()
anomaly = outlier_detector.fit_predict(X)

# select all rows that are not outliers
mask = anomaly != -1
X_new, y_new = X[mask, :], y[mask]
print('- NEW:', X_new.shape, y_new.shape)
```

## One-Class SVM
```python
import pandas as pd
from sklearn.svm import OneClassSVM
from ailever.dataset import SKAPI

# load the dataset
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, 'target'].values.ravel()

print('- ORIGIN: ', X.shape, y.shape)
outlier_detector = OneClassSVM(nu=0.01)
anomaly = outlier_detector.fit_predict(X)

# select all rows that are not outliers
mask = anomaly != -1
X_new, y_new = X[mask, :], y[mask]
print('- NEW:', X_new.shape, y_new.shape)
```


