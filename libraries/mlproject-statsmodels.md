## Regression Analysis
### Univairate
```python
```
### Multivariate
```python
```


---


## Time Series Analysis
### Univairate
#### Endogenous without Exogenous
```python
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from ailever.dataset import SMAPI

df = SMAPI.co2(download=False).rename(columns={'co2':'target'}).asfreq('w-sat').fillna(method='ffill').fillna(method='bfill') # CHECK FREQUENCY, 'W-SAT'
y = df['target']
X = None

model = smt.SARIMAX(y, exog=X, trend='n', order=(1,0,1), seasonal_order=(1,0,1,12), freq='w-sat').fit() # CHECK FREQUENCY, 'H'
model.predict(start=y.index[0], end=y.index[-1], exog=X)
display(model.summary())

# [Residual Analysis]
residual = dict()
residual['data'] = pd.DataFrame()
residual['data']['datetime'] = model.resid[1:].index.year
residual['data']['residual'] = model.resid[1:].values
residual_values = residual['data']['residual']

residual['score'] = dict()
residual['score']['stationarity'] = pd.Series(sm.tsa.stattools.adfuller(residual_values, autolag='BIC')[0:4], index=['statistics', 'p-value', 'used lag', 'used observations']) # Null Hypothesis: The Time-series is non-stationalry 
for key, value in sm.tsa.stattools.adfuller(residual_values)[4].items():
    residual['score']['stationarity']['critical value(%s)'%key] = value
    residual['score']['stationarity']['maximum information criteria'] = sm.tsa.stattools.adfuller(residual_values)[5]
    residual['score']['stationarity'] = pd.DataFrame(residual['score']['stationarity'], columns=['stationarity'])

residual['score']['normality'] = pd.DataFrame([stats.shapiro(residual_values)], index=['normality'], columns=['statistics', 'p-value']).T  # Null Hypothesis: The residuals are normally distributed  
residual['score']['autocorrelation'] = sm.stats.diagnostic.acorr_ljungbox(residual_values, lags=[1,5,10,20,50]).T.rename(index={'lb_stat':'statistics', 'lb_pvalue':'p-value'}) # Null Hypothesis: Autocorrelation is absent
residual['score']['autocorrelation'].columns = ['autocorr(lag1)', 'autocorr(lag5)', 'autocorr(lag10)', 'autocorr(lag20)', 'autocorr(lag50)']
residual_analysis = pd.concat([residual['score']['stationarity'], residual['score']['normality'], residual['score']['autocorrelation']], join='outer', axis=1)
display(residual_analysis)

# [Residual Visualization]
residual['fig'] = plt.figure(figsize=(25,15)); layout = (5,2)
residual_graph = sns.regplot(x='index', y='residual', data=residual['data'].reset_index(), ax=plt.subplot2grid(layout, (0,0)))
residual_graph.set_xticklabels(residual['data']['datetime'][residual_graph.get_xticks()[:-1]])
residual['fig'].add_axes(residual_graph)
residual['fig'].add_axes(sns.histplot(residual_values, kde=True, ax=plt.subplot2grid(layout, (1,0))))
residual['fig'].add_axes(sm.graphics.qqplot(residual_values, dist=stats.norm, fit=True, line='45', ax=plt.subplot2grid(layout, (2,0))).axes[0])
residual['fig'].add_axes(sm.tsa.graphics.plot_acf(residual_values, lags=40, use_vlines=True, ax=plt.subplot2grid(layout, (3,0))).axes[0])
residual['fig'].add_axes(sm.tsa.graphics.plot_pacf(residual_values, lags=40, method='ywm', use_vlines=True, ax=plt.subplot2grid(layout, (4,0))).axes[0])
residual['fig'].add_axes(pd.plotting.lag_plot(residual_values, lag=1, ax=plt.subplot2grid(layout, (0,1))))
residual['fig'].add_axes(pd.plotting.lag_plot(residual_values, lag=5, ax=plt.subplot2grid(layout, (1,1))))
residual['fig'].add_axes(pd.plotting.lag_plot(residual_values, lag=10, ax=plt.subplot2grid(layout, (2,1))))
residual['fig'].add_axes(pd.plotting.lag_plot(residual_values, lag=20, ax=plt.subplot2grid(layout, (3,1))))
residual['fig'].add_axes(pd.plotting.lag_plot(residual_values, lag=50, ax=plt.subplot2grid(layout, (4,1))))
plt.tight_layout()
```
#### Endogenous with Exogenous
```python
```


### Multivariate
#### Endogenous without Exogenous
```python
```

#### Endogenous with Exogenous
```python
```


