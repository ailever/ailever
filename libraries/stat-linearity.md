
## Analysis
### Multi-collinearity
```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ailever.dataset import SMAPI

dataset = SMAPI.macrodata(download=False).rename(columns={'infl':'target'})
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, 'target'].ravel()

VIF = pd.DataFrame()
VIF['VIF_Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
VIF['Feature'] = X.columns
VIF
```
