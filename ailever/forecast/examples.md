

## Time Series Analysis
### Ailever Procedure
#### Case: REITs
```python
import FinanceDataReader as fdr

df = fdr.DataReader('ARE')
df = df.asfreq('B').fillna(method='ffill').fillna(method='bfill')
df
```

### TSA Procedure
#### Case: Beijing Airquality
```python
import re
from datetime import datetime

# preprocessing
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from ailever.dataset import UCI
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion

# modeling
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from prophet import Prophet

# evaluation
from sklearn.model_selection import train_test_split, cross_validate
from sklearn import metrics


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

def predictor():
    def decorator(func):
        def wrapper(model, X, y, model_name='model', domain_kind='train'):
            if model_name == 'SARIMAX' and domain_kind == 'train':
                y_ =  model.predict(start=y.index[0], end=y.index[-1], exog=X)
            elif model_name == 'SARIMAX' and domain_kind == 'test':
                #return = model.get_forecast(y.shape[0], exog=X).predicted_mean
                #return model.get_forecast(y.shape[0], exog=X).conf_int()
                y_ =  model.forecast(steps=y.shape[0], exog=X)
            elif model_name == 'Prophet':
                """ 'ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
                    'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
                    'daily', 'daily_lower', 'daily_upper', 'weekly', 'weekly_lower',
                    'weekly_upper', 'yearly', 'yearly_lower', 'yearly_upper',
                    'multiplicative_terms', 'multiplicative_terms_lower',
                    'multiplicative_terms_upper', 'yhat'"""
                y_ = model.predict(model.make_future_dataframe(freq='H', periods=y.shape[0]))['yhat'].values[:y.shape[0]]
            else:
                y_ = model.predict(X)
                
            if not isinstance(y_, pd.Series):
                y_ = pd.Series(y_, index=y.index)
            return y_
        return wrapper
    return decorator

@predictor()
def prediction(model, X:pd.Series, y:pd.Series, model_name='model', domain_kind='train'):
    return

def residual_analysis(y_true:np.ndarray, y_pred:np.ndarray, date_range:pd.Index, X_true:np.ndarray=None, visual_on=False):
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
    score['heteroscedasticity'] = pd.DataFrame([sm.stats.diagnostic.het_goldfeldquandt(residual_values, X_true, alternative='two-sided')], index=['heteroscedasticity'], columns=['statistics', 'p-value', 'alternative']).T # Null Hypothesis: Error terms are homoscedastic
    residual_eval_matrix = pd.concat([score['stationarity'], score['normality'], score['autocorrelation'], score['heteroscedasticity']], join='outer', axis=1)    
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

def evaluation(y_true, y_pred, date_range, model_name='model', domain_kind='train'):
    summary = dict()
    summary['datetime'] = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    summary['model'] = [model_name]
    summary['domain'] = [domain_kind]
    summary['start'] = [date_range[0].to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')]
    summary['end'] = [date_range[-1].to_pydatetime().strftime('%Y-%m-%d %H:%M:%S')]            
    summary['MAE'] = [metrics.mean_absolute_error(y_true, y_pred)]
    summary['MAPE'] = [metrics.mean_absolute_percentage_error(y_true, y_pred)]
    summary['MSE'] = [metrics.mean_squared_error(y_true, y_pred)]    
    summary['R2'] = [metrics.r2_score(y_true, y_pred)]
    eval_matrix = pd.DataFrame(summary)
    return eval_matrix

print('- PREPROCESSING...')
df = UCI.beijing_airquality(download=False).rename(columns={'pm2.5':'target'})
df['year'] = df.year.astype(str)
df['month'] = df.month.astype(str)
df['day'] = df.day.astype(str)
df['hour'] = df.hour.astype(str)

# [datetime&index preprocessing] time domain seqence integrity
df.index = pd.to_datetime(df.year + '-' + df.month + '-' + df.day + '-' + df.hour, format='%Y-%m-%d-%H')
df = df.asfreq('H').fillna(method='ffill').fillna(method='bfill') # CHECK FREQUENCY, 'H'

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
df['datetime_dayofweek'] = df.index.dayofweek.astype(int)
df['datetime_hourofday'] = df.index.hour.astype(int)

# [endogenous&target feature engineering] decomposition, rolling
decomposition = smt.seasonal_decompose(df['target'], model=['additive', 'multiplicative'][0])
df['target_trend'] = decomposition.trend.fillna(method='ffill').fillna(method='bfill')
df['target_seasonal'] = decomposition.seasonal
df['target_by_day'] = decomposition.observed.rolling(24).mean().fillna(method='ffill').fillna(method='bfill')
df['target_by_week'] = decomposition.observed.rolling(24*7).mean().fillna(method='ffill').fillna(method='bfill')
df['target_by_month'] = decomposition.observed.rolling(24*int(365/12)).mean().fillna(method='ffill').fillna(method='bfill')
df['target_by_quarter'] = decomposition.observed.rolling(24*int(365/4)).mean().fillna(method='ffill').fillna(method='bfill')
df['target_trend_by_day'] = decomposition.trend.rolling(24).mean().fillna(method='ffill').fillna(method='bfill')
df['target_trend_by_week'] = decomposition.trend.rolling(24*7).mean().fillna(method='ffill').fillna(method='bfill')
df['target_trend_by_month'] = decomposition.trend.rolling(24*int(365/12)).mean().fillna(method='ffill').fillna(method='bfill')
df['target_trend_by_quarter'] = decomposition.trend.rolling(24*int(365/4)).mean().fillna(method='ffill').fillna(method='bfill')
df['target_seasonal_by_day'] = decomposition.seasonal.rolling(24).mean().fillna(method='ffill').fillna(method='bfill')
df['target_seasonal_by_week'] = decomposition.seasonal.rolling(24*7).mean().fillna(method='ffill').fillna(method='bfill')
df['target_seasonal_by_month'] = decomposition.seasonal.rolling(24*int(365/12)).mean().fillna(method='ffill').fillna(method='bfill')
df['target_seasonal_by_quarter'] = decomposition.seasonal.rolling(24*int(365/4)).mean().fillna(method='ffill').fillna(method='bfill')

# [exogenous feature engineering] categorical variable to numerical variables
df = pd.concat([df, pd.get_dummies(df['cbwd'], prefix='cbwd')], axis=1).drop('cbwd', axis=1).astype(float)
X = df.loc[:, df.columns != 'target']
y = df.loc[:, df.columns == 'target']

# [exogenous feature engineering] Feature Selection by MultiCollinearity after scaling
fs = FeatureSelection()
X = fs.fit_transform(X)

# [dataset split] Valiation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
y_train_for_prophet = y_train.reset_index()
y_train_for_prophet.columns = ['ds', 'y']
yX_train_for_prophet = pd.concat([y_train_for_prophet, X_train.reset_index().iloc[:,1:]], axis=1)

# [modeling]
print('- MODELING...')
models = dict()
models['OLS'] = sm.OLS(y_train, X_train).fit() #display(models['OLS'].summay())
models['Ridge'] = Ridge(alpha=0.5, fit_intercept=True, normalize=False, random_state=2022).fit(X_train.values, y_train.values.ravel())
models['Lasso'] = Lasso(alpha=0.5, fit_intercept=True, normalize=False, random_state=2022).fit(X_train.values, y_train.values.ravel())
models['ElasticNet'] = ElasticNet(alpha=0.01, l1_ratio=1, fit_intercept=True, normalize=False, random_state=2022).fit(X_train.values, y_train.values.ravel())
models['DecisionTreeRegressor'] = DecisionTreeRegressor().fit(X_train.values, y_train.values.ravel())
models['RandomForestRegressor'] = RandomForestRegressor(n_estimators=100, random_state=2022).fit(X_train.values, y_train.values.ravel())
models['GradientBoostingRegressor'] = GradientBoostingRegressor(alpha=0.1, learning_rate=0.05, loss='huber', criterion='friedman_mse', n_estimators=1000, random_state=2022).fit(X_train.values, y_train.values.ravel())
models['XGBRegressor'] = XGBRegressor(learning_rate=0.05, n_estimators=100, random_state=2022).fit(X_train.values, y_train.values.ravel())
models['LGBMRegressor'] = LGBMRegressor(learning_rate=0.05, n_estimators=100, random_state=2022).fit(X_train.values, y_train.values.ravel())
models['SARIMAX'] = sm.tsa.SARIMAX(y_train, exog=X_train, trend='n', order=(1,1,1), seasonal_order=(1,0,1,12), freq='H').fit() # CHECK FREQUENCY, 'H'
models['Prophet'] = Prophet(growth='linear', changepoints=None, n_changepoints=25, changepoint_range=0.8, changepoint_prior_scale=0.05, 
                            seasonality_mode='additive', seasonality_prior_scale=10.0,  yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality='auto',
                            holidays=None, holidays_prior_scale=10.0, 
                            interval_width=0.8, mcmc_samples=0).fit(yX_train_for_prophet)

order = 1 + 1*12 + 1 + 0 # p + P*m + d + D*m
y_train_true = y_train[order:]  # pd.Sereis
y_test_true = y_test[order:]    # pd.Sereis
X_train_true = X_train[order:]  # pd.Sereis
X_test_true = X_test[order:]    # pd.Sereis

y.plot(lw=0, marker='o', c='black', grid=True, figsize=(25,7))
print('- EVALUATIING...')
for idx, (name, model) in enumerate(models.items()):
    print(name)
    y_train_pred = prediction(model, X_train, y_train, model_name=name, domain_kind='train')[order:] # pd.Series
    y_test_pred = prediction(model, X_test, y_test, model_name=name, domain_kind='test')[order:]     # pd.Series
    
    # Visualization Process
    pd.Series(data=y_train_pred.values.squeeze(), index=y_train.index[order:], name=name+'|train').plot(legend=True, grid=True, figsize=(25,7))
    pd.Series(data=y_test_pred.values.squeeze(), index=y_test.index[order:], name=name+'|test').plot(legend=True, grid=True, figsize=(25,7))
    
    # Evaluation Process
    residual_eval_matrix = residual_analysis(y_train_true.values.squeeze(), y_train_pred.values.squeeze(), date_range=y_train.index[order:], X_true=X_train_true.values.squeeze(), visual_on=False)
    eval_matrix = evaluation(y_train_true.values.squeeze(), y_train_pred.values.squeeze(), date_range=y_train.index[order:], model_name=name, domain_kind='train')
    eval_matrix = pd.concat([eval_matrix, residual_eval_matrix.loc['judgement'].rename(0).to_frame().astype(bool).T], axis=1)
    if idx == 0:
        eval_table = eval_matrix.copy() 
    else:
        eval_table = eval_table.append(eval_matrix.copy()) 
        
    residual_eval_matrix = residual_analysis(y_test_true.values.squeeze(), y_test_pred.values.squeeze(), date_range=y_test.index[order:], X_true=X_test_true.values.squeeze(), visual_on=False)
    eval_matrix = evaluation(y_test_true.values.squeeze(), y_test_pred.values.squeeze(), date_range=y_test.index[order:], model_name=name, domain_kind='test')
    eval_matrix = pd.concat([eval_matrix, residual_eval_matrix.loc['judgement'].rename(0).to_frame().astype(bool).T], axis=1)    
    eval_table = eval_table.append(eval_matrix.copy())
display(eval_table)

# [Data Analysis]
display(df.groupby(['datetime_monthofyear', 'datetime_dayofmonth']).describe().T)

condition = df.loc[lambda x: x.datetime_dayofmonth == 30, :]
condition_table = pd.crosstab(index=condition['target'], columns=condition['datetime_monthofyear'], margins=True)
condition_table = condition_table/condition_table.loc['All']*100

# [Data Visualization]
display(condition.describe(percentiles=[ 0.1*i for i in range(1, 10)], include='all').T)
display(condition.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'}))
condition.hist(bins=30, grid=True, figsize=(27,12))
condition.boxplot(column='target', by='datetime_monthofyear', grid=True, figsize=(25,5))
condition.plot.scatter(y='target',  x='datetime_monthofyear', c='TEMP', grid=True, figsize=(25,5), colormap='viridis', colorbar=True)
plt.tight_layout()
```


