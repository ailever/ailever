### Ordinary Least Squares
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from ailever.dataset import SMAPI

df = SMAPI.macrodata(download=False)
df.index = pd.date_range(start='1959-01-01', periods=df.shape[0], freq='Q')

time = np.linspace(-5, 5, df.shape[0])
realgdp = df['realgdp'].values
X = np.c_[np.ones(df.shape[0]), time, realgdp]

model = sm.OLS(df['realint'], X)
model = model.fit()
model.summary()
```

### Simple Exponential Smoothing
```python
import pandas as pd
import statsmodels.api as sm
from ailever.dataset import SMAPI

df = SMAPI.macrodata(download=False)
df.index = pd.date_range(start='1959-01-01', periods=df.shape[0], freq='Q')

model = sm.tsa.SimpleExpSmoothing(df['realint'])
model = model.fit(smoothing_level=0.9, optimized=True, remove_bias=False, method='L-BFGS-B')
model.summary()
```

### Holt's Exponential Smoothing
```python
import pandas as pd
import statsmodels.api as sm
from ailever.dataset import SMAPI

df = SMAPI.macrodata(download=False)
df.index = pd.date_range(start='1959-01-01', periods=df.shape[0], freq='Q')

model = sm.tsa.Holt(df['unemp'], exponential=True, damped_trend=True)
model = model.fit()
model.summary()
```


### Holt Winter's Exponential Smoothing
```python
import pandas as pd
import statsmodels.api as sm
from ailever.dataset import SMAPI

df = SMAPI.macrodata(download=False)
df.index = pd.date_range(start='1959-01-01', periods=df.shape[0], freq='Q')

model = sm.tsa.ExponentialSmoothing(df['unemp'], seasonal_periods=4, trend='add', seasonal='add', damped_trend=False)
model = model.fit()
model.summary()
```

### ETS Model
```python
import pandas as pd
import statsmodels.api as sm
from ailever.dataset import SMAPI

df = SMAPI.macrodata(download=False)
df.index = pd.date_range(start='1959-01-01', periods=df.shape[0], freq='Q')

model = sm.tsa.ETSModel(df['unemp'], seasonal_periods=4, error='add', trend=None, seasonal=None, damped_trend=False)
model = model.fit()
model.summary()
```




--- 

### SARIMAX
`with exogenous`
```python
import pandas as pd
import statsmodels.api as sm
from ailever.dataset import SMAPI

df = SMAPI.macrodata(download=False)
df.index = pd.date_range(start='1959-01-01', periods=df.shape[0], freq='Q')

model = sm.tsa.statespace.SARIMAX(endog=df.realint, exog=sm.add_constant(df.loc[:, 'realgdp':'infl']), order=(2,1,0), seasonal_order=(1,0,1,4), freq='Q')
model = model.fit(disp=False, method='lbfgs', maxiter=200)
model.summary()
```

`with trend`
```python
import pandas as pd
import statsmodels.api as sm
from ailever.dataset import SMAPI

df = SMAPI.macrodata(download=False)
df.index = pd.date_range(start='1959-01-01', periods=df.shape[0], freq='Q')

model = sm.tsa.statespace.SARIMAX(endog=df.realint, trend='c', order=(2,1,0), seasonal_order=(1,0,1,4), freq='Q')
model = model.fit(disp=False, method='lbfgs', maxiter=200)
model.summary()
```

### VARMAX
```python
import pandas as pd
import statsmodels.api as sm
from ailever.dataset import SMAPI

df = SMAPI.macrodata(download=False)
df.index = pd.date_range(start='1959-01-01', periods=df.shape[0], freq='Q')

model = sm.tsa.VARMAX(df[['infl', 'realint']], exog=df.loc[:, 'realgdp':'pop'], order=(2,1), trend='n')
model = model.fit(disp=False, method='lbfgs', maxiter=200)
model.summary()
```
