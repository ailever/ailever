## [Machine Learning] | [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_api.html) | [GitHub](https://github.com/dmlc/xgboost)

## Classification
```python
import joblib
import xgboost as xgb
from ailever.dataset import SKAPI

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
classifier = xgb.XGBClassifier(objective='multi:softprob')
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib.dat')
classifier = joblib.load('classifier.joblib.dat')

# [STEP4]: prediction
classifier.predict(X[0:10])
```

## Regression
```python
import joblib
import xgboost as xgb
from ailever.dataset import SKAPI

# [STEP1]: data
dataset = SKAPI.boston(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib.dat')
regressor = joblib.load('regressor.joblib.dat')

# [STEP4]: prediction
regressor.predict(X[0:10])
```