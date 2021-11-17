## [Machine Learning] | [LightGBM](https://lightgbm.readthedocs.io/en/latest/Python-API.html) | [GitHub](https://github.com/microsoft/LightGBM)

## Classfication
```python
import joblib
import lightgbm as lgb
from ailever.dataset import SKAPI

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = lgb.LGBMClassifier(objective="multiclass")
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```

## Regression
```python
import joblib
import lightgbm as lgb
from ailever.dataset import SKAPI

# [STEP1]: data
dataset = SKAPI.boston(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = lgb.LGBMRegressor(objective="regression")
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```

