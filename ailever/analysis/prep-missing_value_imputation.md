
## KNNImputer
```python
import numpy as np
from sklearn.impute import KNNImputer

X = [[1, 2, np.nan], 
     [3, 4, 3], 
     [np.nan, 6, 5], 
     [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2)
X_new = imputer.fit_transform(X)
X_new
```

## SimpleImputer: Statistical Imputation
```python
import numpy as np
from sklearn.impute import SimpleImputer

X = [[1, 2, np.nan], 
     [3, 4, 3], 
     [np.nan, 6, 5], 
     [8, 8, 7]]
imputer = SimpleImputer()
X_new = imputer.fit_transform(X)
X_new
```
