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
