
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.beijing_airquality(download=False)
df['year'] = df.year.astype(str)
df['month'] = df.month.astype(str)
df['day'] = df.day.astype(str)
df['hour'] = df.hour.astype(str)

# time domain seqence integrity
df.index = pd.to_datetime(df.year + '-' + df.month + '-' + df.day + '-' + df.hour, format='%Y-%m-%d-%H')
df = df.asfreq('H').fillna(method='ffill').fillna(method='bfill')
df['year'] = df.index.year.astype(int)
df['quarterofyear'] = df.index.quarter.astype(int)
df['monthofyear'] = df.index.month.astype(int)
df['weekofyear'] = df.index.isocalendar().week # week of year
df['dayofyear'] = df.index.dayofyear

df['day'] = df.index.day.astype(int)
df['hour'] = df.index.hour.astype(int)

# categorical variable to numerical variables
df = pd.concat([df, pd.get_dummies(df['cbwd'], prefix='cbwd')], axis=1).drop('cbwd', axis=1)
```
