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
from sklearn.model_selection import train_test_split, cross_validate
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

# [dataset split] Valiation
y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)

# [Modeling]
model = smt.SARIMAX(y_train, exog=X, trend='c', order=(4,1,2), seasonal_order=(2,0,1,5), freq='w-sat').fit() # CHECK FREQUENCY, 'w-sat'
display(model.summary())

# [Residual Analysis] 
# y_resid = model.resid.values
order = 4 + 2*5 + 1 + 0 # p + P*m + d + D*m
y_true = y_train[order:].values.squeeze()
y_pred = model.predict(start=y_train.index[0], end=y_train.index[-1], exog=X)[order:].values

residual_eval_matrix = residual_analysis(y_true, y_pred, date_range=y_train.index[order:], visual_on=True)
display(residual_eval_matrix)

eval_table = evaluation(y_true, y_pred, model_name='SARIMAX', domain_kind='train')
eval_table = pd.concat([eval_table, residual_eval_matrix.loc['judgement'].rename(0).to_frame().astype(bool).T], axis=1)
display(eval_table)

# [Inference]
fig = plt.figure(figsize=(25,7))
ax = plt.subplot2grid((1,1), (0,0))
fig.add_axes(y[order:].plot(lw=0, marker='o', c='black', ax=ax))
fig.add_axes(model.predict(start=y_train.index[order], end=y_train.index[-1], exog=X).plot(grid=True, ax=ax))
fig.add_axes(model.forecast(steps=y_test.shape[0], exog=X).plot(grid=True, ax=ax))
```
```python
residual_analysis(y.diff(1).dropna().values, 0, df.index[1:], visual_on=True)
```

#### Endogenous with Exogenous
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import metrics
from ailever.dataset import SMAPI

class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import pandas as pd
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from sklearn.preprocessing import Normalizer # MinMaxScaler, StandardScaler, RobustScaler, Normalizer

        # Scaling
        X = pd.DataFrame(data=Normalizer().fit_transform(X), index=X.index, columns=X.columns)
        
        # VIF Feature Selection
        features_by_vif = pd.Series(
            data = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])], 
            index = range(X.shape[1])).sort_values(ascending=True).iloc[:X.shape[1] - 1].index.tolist()
        return X.iloc[:, features_by_vif].copy()

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

# [time series core feature] previous time series
df['target_lag24'] = df['target'].shift(24).fillna(method='bfill')
df['target_lag48'] = df['target'].shift(48).fillna(method='bfill')
df['target_lag72'] = df['target'].shift(72).fillna(method='bfill')
df['target_lag96'] = df['target'].shift(96).fillna(method='bfill')
df['target_lag120'] = df['target'].shift(120).fillna(method='bfill')

# [time series core feature] current time series properties
df['datetime_year'] = df.index.year.astype(int)
df['datetime_quarterofyear'] = df.index.quarter.astype(int)
df['datetime_monthofyear'] = df.index.month.astype(int)
df['datetime_weekofyear'] = df.index.isocalendar().week # week of year
df['datetime_dayofyear'] = df.index.dayofyear
df['datetime_dayofmonth'] = df.index.day.astype(int)

# [endogenous&target feature engineering] decomposition, rolling
decomposition = smt.seasonal_decompose(df['target'], model=['additive', 'multiplicative'][0])
df['target_trend'] = decomposition.trend.fillna(method='ffill').fillna(method='bfill')
df['target_seasonal'] = decomposition.seasonal
df['target_by_month'] = decomposition.seasonal.rolling(4).mean().fillna(method='ffill').fillna(method='bfill')
df['target_by_quarter'] = decomposition.seasonal.rolling(4*3).mean().fillna(method='ffill').fillna(method='bfill')

# [exogenous feature engineering] split
X = df.loc[:, df.columns != 'target']
y = df.loc[:, df.columns == 'target']

# [exogenous feature engineering] Feature Selection by MultiCollinearity after scaling
fs = FeatureSelection()
X = fs.fit_transform(X)

# [dataset split] Valiation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# [Modeling]
model = smt.SARIMAX(y_train, exog=X_train, trend='c', order=(4,1,2), seasonal_order=(2,0,1,5), freq='w-sat').fit() # CHECK FREQUENCY, 'w-sat'
display(model.summary())

# [Residual Analysis] 
# y_resid = model.resid.values
order = 4 + 2*5 + 1 + 0 # p + P*m + d + D*m
y_true = y_train[order:].values.squeeze()
y_pred = model.predict(start=y_train.index[0], end=y_train.index[-1], exog=X_train)[order:].values

residual_eval_matrix = residual_analysis(y_true, y_pred, date_range=y_train.index[order:], visual_on=True)
display(residual_eval_matrix)

eval_table = evaluation(y_true, y_pred, model_name='SARIMAX', domain_kind='train')
eval_table = pd.concat([eval_table, residual_eval_matrix.loc['judgement'].rename(0).to_frame().astype(bool).T], axis=1)
display(eval_table)

# [Inference]
fig = plt.figure(figsize=(25,7))
ax = plt.subplot2grid((1,1), (0,0))
fig.add_axes(y[order:].plot(lw=0, marker='o', c='black', ax=ax))
fig.add_axes(model.predict(start=y_train.index[order], end=y_train.index[-1], exog=X_train[order:]).plot(grid=True, ax=ax))
fig.add_axes(model.forecast(steps=X_test.shape[0], exog=X_test).plot(grid=True, ax=ax))
```


### Multivariate
#### Endogenous without Exogenous
```python
```

#### Endogenous with Exogenous
```python
```


## ARIMA Utils
### AutoCorrelation
```python
import numpy as np
import statsmodels.api as sm

np.random.seed(12345)
arparams = np.array([.01, -.01])
maparams = np.array([.01, .01])
ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1, maparams] # add zero-lag
residual_values = sm.tsa.arma_generate_sample(ar, ma, 250)

score = dict()
score['autocorrelation'] = sm.stats.diagnostic.acorr_ljungbox(residual_values, lags=[1,2,3,4,5]).T.rename(index={'lb_stat':'statistics', 'lb_pvalue':'p-value'}) # Null Hypothesis: Autocorrelation is absent
score['autocorrelation'].columns = ['autocorr(lag1)', 'autocorr(lag5)', 'autocorr(lag10)', 'autocorr(lag20)', 'autocorr(lag50)']
score['autocorrelation']
```

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



