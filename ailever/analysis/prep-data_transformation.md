
## Gaussian-like Transform
### yeo-johnson transform
```python
from ailever.dataset import SMAPI
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import PowerTransformer

# [origin]
frame = SMAPI.macrodata(download=False) # Multivariate Time Series Dataset
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [transform]
pt = PowerTransformer(method='yeo-johnson') 
transformed_frame = pt.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```

### box-cox transform(for positive-data)
```python
import pandas as pd
from pandas.plotting import scatter_matrix
import statsmodels.tsa.api as smt
from sklearn.preprocessing import PowerTransformer
from ailever.dataset import SMAPI

frame = SMAPI.co2(download=False).rename(columns={'co2':'target'}).asfreq('w-sat').fillna(method='ffill').fillna(method='bfill') # CHECK FREQUENCY, 'W-SAT'
frame['target_lag24'] = frame['target'].shift(24).fillna(method='bfill')
frame['target_lag48'] = frame['target'].shift(48).fillna(method='bfill')
frame['target_lag72'] = frame['target'].shift(72).fillna(method='bfill')
frame['target_lag96'] = frame['target'].shift(96).fillna(method='bfill')
frame['target_lag120'] = frame['target'].shift(120).fillna(method='bfill')
frame['datetime_year'] = frame.index.year.astype(int)
frame['datetime_quarterofyear'] = frame.index.quarter.astype(int)
frame['datetime_monthofyear'] = frame.index.month.astype(int)
frame['datetime_weekofyear'] = frame.index.isocalendar().week # week of year
frame['datetime_dayofyear'] = frame.index.dayofyear
frame['datetime_dayofmonth'] = frame.index.day.astype(int)
decomposition = smt.seasonal_decompose(frame['target'], model=['additive', 'multiplicative'][0])
frame['target_trend'] = decomposition.trend.fillna(method='ffill').fillna(method='bfill')
frame['target_trend_by_month'] = decomposition.trend.rolling(4).mean().fillna(method='ffill').fillna(method='bfill')
frame['target_trend_by_quarter'] = decomposition.trend.rolling(4*3).mean().fillna(method='ffill').fillna(method='bfill')
#frame['target_seasonal'] = decomposition.seasonal
#frame['target_seasonal_by_month'] = decomposition.seasonal.rolling(4).mean().fillna(method='ffill').fillna(method='bfill')
#frame['target_seasonal_by_quarter'] = decomposition.seasonal.rolling(4*3).mean().fillna(method='ffill').fillna(method='bfill')

# [origin]
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [transform]
pt = PowerTransformer(method='box-cox') 
transformed_frame = pt.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```
