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

def residual_analysis(y_true, y_pred, date_range, visual_on=False):
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
model = smt.SARIMAX(y, exog=X, trend='t', order=(1,1,1), seasonal_order=(1,0,1,12), freq='w-sat').fit() # CHECK FREQUENCY, 'H'
display(model.summary())

# [Residual Analysis] 
# y_resid = model.resid.values
y_true = y[1:].values
y_pred = model.predict(start=y.index[0], end=y.index[-1], exog=X)[1:].values

residual_eval_matrix = residual_analysis(y_true, y_pred, date_range=y.index[1:], visual_on=True)
display(residual_eval_matrix)

eval_table = evaluation(y_true, y_pred, model_name='SARIMAX', domain_kind='train')
eval_table = pd.concat([eval_table, residual_eval_matrix.loc['judgement'].rename(0).to_frame().astype(bool).T], axis=1)
display(eval_table)

# [Inference]
fig = plt.figure(figsize=(25,7))
fig.add_axes(model.predict(start=y.index[1], end=y.index[-1]).plot(grid=True))
fig.add_axes(model.forecast(steps=300).plot(grid=True))
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


