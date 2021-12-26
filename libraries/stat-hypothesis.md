## Time Series Analysis
### Normality Testing
### AutoCorrelation Testing
### Heteroscedasticity Testing
### Stationary Testing
### Unit Root Testing
```python
from arch.unitroot import ADF, DFGLS, PhillipsPerron, ZivotAndrews, VarianceRatio, KPSS
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
unit_root_testing['dfgls'] = DFGLS(y)
unit_root_testing['pp'] = PhillipsPerron(y)
unit_root_testing['za'] = ZivotAndrews(y)
unit_root_testing['vr'] = VarianceRatio(y)
unit_root_testing['kpss'] = KPSS(y)

#unit_root_testing['~'].trend
#unit_root_testing['~'].lags
#unit_root_testing['~'].stat
#unit_root_testing['~'].pvalue
```

### Cointegration Testing
```python
from arch.unitroot import cointegration
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as smt

n_samples = 300
ar_params = np.r_[0.3, 0.1]
ma_params = np.r_[0.1, 0.1]
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]

y1 = smt.ArmaProcess(ar, ma).generate_sample(n_samples, burnin=50)
y2 = 0.3 * y + np.random.normal(size=n_samples)  # schtochastic/deterministic

cointegration_testing = dict()
cointegration_testing['eg'] = cointegration.engle_granger(y1, y2)
cointegration_testing['po'] = cointegration.phillips_ouliaris(y1, y2)

#cointegration_testing['~'].trend
#cointegration_testing['~'].stat
#cointegration_testing['~'].pvalue
```


