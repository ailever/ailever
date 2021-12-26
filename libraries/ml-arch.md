## Hypothesis Test
### Unit Root Testing
```python
from arch.unitroot import ADF
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as smt

n_samples = 300
ar_params = np.r_[0.3, 0.1]
ma_params = np.r_[0.1, 0.1]
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]

y = smt.ArmaProcess(ar, ma).generate_sample(n_samples, burnin=50)

unit_root_testing = dict()
unit_root_testing['adf'] = ADF(y)


unit_root_testing['adf'].trend
unit_root_testing['adf'].lags
unit_root_testing['adf'].stat
unit_root_testing['adf'].pvalue


```
### Cointegration Testing


