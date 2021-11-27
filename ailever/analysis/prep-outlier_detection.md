## Isolation Forest
```python
# evaluate model performance with outliers removed using isolation forest
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
```

## Local Outlier Factor
```python
```

## One-Class SVM
```python
```


