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
from datetime import datetime
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn import metrics
from ailever.dataset import SMAPI

def residual_analysis(y_true:np.ndarray, y_pred:np.ndarray, date_range:pd.Index, visual_on=False):
    residual = pd.DataFrame()
    residual['datetime'] = date_range
    residual['residual'] = y_true - y_pred
    residual_values = residual['residual']

    score = dict()
    score['stationarity'] = pd.Series(sm.tsa.stattools.adfuller(residual_values, autolag='BIC')[0:4], index=['statistics', 'p-value', 'used lag', 'used observations']) # Null Hypothesis: The Time-series is non-stationalry 
    for key, value in sm.tsa.stattools.adfuller(residual_values)[4].items():
        score['stationarity']['critical value(%s)'%key] = value
        score['stationarity']['maximum information criteria'] = sm.tsa.stattools.adfuller(residual_values)[5]
        score['stationarity'] = pd.DataFrame(score['stationarity'], columns=['stationarity'])

    score['normality'] = pd.DataFrame([stats.shapiro(residual_values)], index=['normality'], columns=['statistics', 'p-value']).T  # Null Hypothesis: The residuals are normally distributed  
    score['autocorrelation'] = sm.stats.diagnostic.acorr_ljungbox(residual_values, lags=[1,5,10,20,50]).T.rename(index={'lb_stat':'statistics', 'lb_pvalue':'p-value'}) # Null Hypothesis: Autocorrelation is absent
    score['autocorrelation'].columns = ['autocorr(lag1)', 'autocorr(lag5)', 'autocorr(lag10)', 'autocorr(lag20)', 'autocorr(lag50)']
    residual_eval_matrix = pd.concat([score['stationarity'], score['normality'], score['autocorrelation']], join='outer', axis=1)
    residual_eval_matrix = residual_eval_matrix.append(residual_eval_matrix.T['p-value'].apply(lambda x: True if x < 0.05 else False).rename('judgement'))
        
    if visual_on:
        fig = plt.figure(figsize=(25,15)); layout = (5,2)
        residual_graph = sns.regplot(x='index', y='residual', data=residual.reset_index(), ax=plt.subplot2grid(layout, (0,0)))
        xticks = residual_graph.get_xticks()[:-1].tolist()
        residual_graph.xaxis.set_major_locator(mticker.FixedLocator(xticks))
        residual_graph.set_xticklabels(residual['datetime'][xticks])

        fig.add_axes(residual_graph)
        fig.add_axes(sns.histplot(residual_values, kde=True, ax=plt.subplot2grid(layout, (1,0))))
        fig.add_axes(sm.graphics.qqplot(residual_values, dist=stats.norm, fit=True, line='45', ax=plt.subplot2grid(layout, (2,0))).axes[0])
        fig.add_axes(sm.tsa.graphics.plot_acf(residual_values, lags=40, use_vlines=True, ax=plt.subplot2grid(layout, (3,0))).axes[0])
        fig.add_axes(sm.tsa.graphics.plot_pacf(residual_values, lags=40, method='ywm', use_vlines=True, ax=plt.subplot2grid(layout, (4,0))).axes[0])
        fig.add_axes(pd.plotting.lag_plot(residual_values, lag=1, ax=plt.subplot2grid(layout, (0,1))))
        fig.add_axes(pd.plotting.lag_plot(residual_values, lag=5, ax=plt.subplot2grid(layout, (1,1))))
        fig.add_axes(pd.plotting.lag_plot(residual_values, lag=10, ax=plt.subplot2grid(layout, (2,1))))
        fig.add_axes(pd.plotting.lag_plot(residual_values, lag=20, ax=plt.subplot2grid(layout, (3,1))))
        fig.add_axes(pd.plotting.lag_plot(residual_values, lag=50, ax=plt.subplot2grid(layout, (4,1))))
        plt.tight_layout()
        
    return residual_eval_matrix

def evaluation(y_true, y_pred, model_name='model', domain_kind='train'):
    summary = dict()
    summary['datetime'] = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    summary['model'] = [model_name]
    summary['domain'] = [domain_kind]
    summary['MAE'] = [metrics.mean_absolute_error(y_true, y_pred)]
    summary['MAPE'] = [metrics.mean_absolute_percentage_error(y_true, y_pred)]
    summary['MSE'] = [metrics.mean_squared_error(y_true, y_pred)]    
    summary['R2'] = [metrics.r2_score(y_true, y_pred)]
    eval_matrix = pd.DataFrame(summary)
    return eval_matrix

# [Preprocessing]
df = SMAPI.co2(download=False).rename(columns={'co2':'target'}).asfreq('w-sat').fillna(method='ffill').fillna(method='bfill') # CHECK FREQUENCY, 'W-SAT'
y = df['target']
X = None

# [Modeling]
model = smt.SARIMAX(y, exog=X, trend='c', order=(3,1,0), seasonal_order=(1,1,1,12), freq='w-sat').fit() # CHECK FREQUENCY, 'H'
display(model.summary())

# [Residual Analysis] 
# y_resid = model.resid.values
order = 3 + 1*12 + 1 + 1*12 # p + P*m + d + D*m
y_true = y[order:].values
y_pred = model.predict(start=y.index[0], end=y.index[-1], exog=X)[order:].values

residual_eval_matrix = residual_analysis(y_true, y_pred, date_range=y.index[order:], visual_on=True)
display(residual_eval_matrix)

eval_table = evaluation(y_true, y_pred, model_name='SARIMAX', domain_kind='train')
eval_table = pd.concat([eval_table, residual_eval_matrix.loc['judgement'].rename(0).to_frame().astype(bool).T], axis=1)
display(eval_table)

# [Inference]
fig = plt.figure(figsize=(25,7))
fig.add_axes(y[order:].plot(lw=0, marker='o', c='black'))
fig.add_axes(model.predict(start=y.index[order], end=y.index[-1]).plot(grid=True))
fig.add_axes(model.forecast(steps=300).plot(grid=True))
```
```python
residual_analysis(y.diff(1).dropna().values, 0, df.index[1:], visual_on=True)
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


## ARIMA Utils
### Order-searching
```python
import pmdarima as pm
from ailever.dataset import SMAPI

df = SMAPI.co2(download=False).rename(columns={'co2':'target'}).asfreq('w-sat').fillna(method='ffill').fillna(method='bfill') # CHECK FREQUENCY, 'W-SAT'
y = df['target']
X = None

# max_order: p+q+P+Q
autoarima = pm.auto_arima(
    y, exogenous=X, maxiter=50, information_criterion='bic', trace=True, suppress_warnings=True,
    stationary=False, with_intercept=True, seasonal=True, m=12, max_order=8, 
    start_p=1, d=1, start_q=1, start_P=1, D=1, start_Q=1, 
    max_p=3, max_d=1, max_q=3, max_P=2, max_D=1, max_Q=2)
```

### De-Trending
```python
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.api as smt
from ailever.dataset import SMAPI

df = SMAPI.co2(download=False).rename(columns={'co2':'target'}).asfreq('w-sat').fillna(method='ffill').fillna(method='bfill') # CHECK FREQUENCY, 'W-SAT'
y = df['target']
X = None

decomposed_series = dict()
decomposed_series['STL'] = STL(y).fit()
decomposed_series['MV'] = smt.seasonal_decompose(y, model='additive')

decomposed_series['STL'].observed - decomposed_series['STL'].trend
decomposed_series['MV'].observed - decomposed_series['MV'].trend
```

### De-Seasonalizing
```python
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.api as smt
from ailever.dataset import SMAPI

df = SMAPI.co2(download=False).rename(columns={'co2':'target'}).asfreq('w-sat').fillna(method='ffill').fillna(method='bfill') # CHECK FREQUENCY, 'W-SAT'
y = df['target']
X = None

decomposed_series = dict()
decomposed_series['STL'] = STL(y).fit()
decomposed_series['MV'] = smt.seasonal_decompose(y, model='additive')

decomposed_series['STL'].observed - decomposed_series['STL'].seasonal
decomposed_series['MV'].observed - decomposed_series['MV'].seasonal
```



