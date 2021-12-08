
## Coefficients as Feature Importance
### Classifier
`Logistic Regression`
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
classifier = LogisticRegression()
classifier.fit(X, y)

# get importance
importance = classifier.coef_[0]

# plot feature importance
plt.barh([x for x in range(len(importance))], importance)
plt.show()

pd.DataFrame(data=importance, columns=['FeatureImportance'])
```
### Regressor
`Linear Regression`
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
regressor = LinearRegression()
regressor.fit(X, y)

# get importance
importance = regressor.coef_

# plot feature importance
plt.barh([x for x in range(len(importance))], importance)
plt.show()

pd.DataFrame(data=importance, columns=['FeatureImportance'])
```

## Decision Tree Feature Importance
### Classifier
`CART`
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
classifier = DecisionTreeClassifier()
classifier.fit(X, y)

# get importance
importance = classifier.feature_importances_

# plot feature importance
plt.barh([x for x in range(len(importance))], importance)
plt.show()

pd.DataFrame(data=importance, columns=['FeatureImportance'])
```
`Random Forest`
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
classifier = RandomForestClassifier()
classifier.fit(X, y)

# get importance
importance = classifier.feature_importances_

# plot feature importance
plt.barh([x for x in range(len(importance))], importance)
plt.show()

pd.DataFrame(data=importance, columns=['FeatureImportance'])
```
`XGBoost`
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
classifier = XGBClassifier(use_label_encoder=False)
classifier.fit(X, y)

# get importance
importance = classifier.feature_importances_

# plot feature importance
plt.barh([x for x in range(len(importance))], importance)
plt.show()

pd.DataFrame(data=importance, columns=['FeatureImportance'])
```

### Regressor
`CART`
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# get importance
importance = regressor.feature_importances_

# plot feature importance
plt.barh([x for x in range(len(importance))], importance)
plt.show()

pd.DataFrame(data=importance, columns=['FeatureImportance'])
```
`Random Forest`
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
regressor = RandomForestRegressor()
regressor.fit(X, y)

# get importance
importance = regressor.feature_importances_

# plot feature importance
plt.barh([x for x in range(len(importance))], importance)
plt.show()

pd.DataFrame(data=importance, columns=['FeatureImportance'])
```
`XGBoost`
```python
# linear regression feature importance
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from xgboost import XGBRegressor

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
regressor = XGBRegressor()
regressor.fit(X, y)

# get importance
importance = regressor.feature_importances_

# plot feature importance
plt.barh([x for x in range(len(importance))], importance)
plt.show()

pd.DataFrame(data=importance, columns=['FeatureImportance'])
```

## Permutation Feature Importance
### Classifier
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.inspection import permutation_importance
import xgboost as xgb

# define dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, random_state=1)
classifier = xgb.XGBClassifier(use_label_encoder=False)
classifier.fit(X, y)

# get importance
importance = permutation_importance(classifier, X, y, scoring='accuracy').importances_mean

# plot permutation importance
plt.barh([x for x in range(len(importance))], importance)
plt.show()

pd.DataFrame(data=importance, columns=['PermutationImportance'])
```
### Regressor
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.inspection import permutation_importance

# define dataset
X, y = make_regression(n_samples=1000, n_features=10, n_informative=5, random_state=1)
regressor = xgb.XGBRegressor()
regressor.fit(X, y)

# get importance
importance = permutation_importance(regressor, X, y, scoring='neg_mean_squared_error').importances_mean

# plot permutation importance
plt.barh([x for x in range(len(importance))], importance)
plt.show()

pd.DataFrame(data=importance, columns=['PermutationImportance'])
```

