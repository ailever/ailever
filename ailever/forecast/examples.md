

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
import pandas as pd
import statsmodels.tsa.api as smt
from ailever.dataset import UCI

df = UCI.beijing_airquality(download=False).rename(columns={'pm2.5':'target'})
df['year'] = df.year.astype(str)
df['month'] = df.month.astype(str)
df['day'] = df.day.astype(str)
df['hour'] = df.hour.astype(str)

# [datetime] time domain seqence integrity
df.index = pd.to_datetime(df.year + '-' + df.month + '-' + df.day + '-' + df.hour, format='%Y-%m-%d-%H')
df = df.asfreq('H').fillna(method='ffill').fillna(method='bfill')
df['datetime_year'] = df.index.year.astype(int)
df['datetime_quarterofyear'] = df.index.quarter.astype(int)
df['datetime_monthofyear'] = df.index.month.astype(int)
df['datetime_weekofyear'] = df.index.isocalendar().week # week of year
df['datetime_dayofyear'] = df.index.dayofyear
df['datetime_dayofmonth'] = df.index.day.astype(int)
df['datetime_dayofweek'] = df.index.dayofweek.astype(int)
df['datetime_hourofday'] = df.index.hour.astype(int)

# [target] decomposition, rolling
decomposition = smt.seasonal_decompose(df['target'], model=['additive', 'multiplicative'][0])
df['target_trend'] = decomposition.trend.fillna(method='ffill').fillna(method='bfill')
df['target_seasonal'] = decomposition.seasonal
df['target_by_day'] = decomposition.observed.rolling(24).mean().fillna(method='ffill').fillna(method='bfill')
df['target_by_week'] = decomposition.seasonal.rolling(24*7).mean().fillna(method='ffill').fillna(method='bfill')
df['target_by_month'] = decomposition.seasonal.rolling(24*int(365/12)).mean().fillna(method='ffill').fillna(method='bfill')
df['target_by_quarter'] = decomposition.seasonal.rolling(24*int(365/4)).mean().fillna(method='ffill').fillna(method='bfill')
df['target_lag24'] = df['target'].shift(24).fillna(method='bfill')
df['target_lag48'] = df['target'].shift(48).fillna(method='bfill')
df['target_lag72'] = df['target'].shift(72).fillna(method='bfill')
df['target_lag96'] = df['target'].shift(96).fillna(method='bfill')
df['target_lag120'] = df['target'].shift(120).fillna(method='bfill')

# categorical variable to numerical variables
df = pd.concat([df, pd.get_dummies(df['cbwd'], prefix='cbwd')], axis=1).drop('cbwd', axis=1)

# reference
#df.groupby(['datetime_monthofyear', 'datetime_dayofmonth']).describe().T

condition = df.loc[lambda x: x.datetime_dayofmonth == 30, :]
condition_table = pd.crosstab(index=condition['target'], columns=condition['datetime_monthofyear'], margins=True)
condition_table = condition_table/condition_table.loc['All']*100

condition.describe(percentiles=[ 0.1*i for i in range(1, 10)], include='all').T
condition.hist(bins=30, grid=True, figsize=(27,12))
condition.boxplot(column='target', by='datetime_monthofyear', grid=True, figsize=(25,5))
condition.plot.scatter(y='target',  x='datetime_monthofyear', c='TEMP', grid=True, figsize=(25,5), colormap='viridis', colorbar=True)
condition.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```


