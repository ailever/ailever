## [Data Analysis] | [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/index.html) | [github](https://github.com/pandas-dev/pandas) | [MDIS](https://mdis.kostat.go.kr/index.do)

### Replace
```python
import pandas as pd
from ailever.analysis import EDA
from ailever.dataset import UCI

DF = EDA(UCI.adult(download=False), verbose=False).cleaning()

# objective : replace instances having education property with Bachelors in this dataframe as an instance 'ABC' of the column 'education' 
df = DF.copy()
df = df.loc[lambda x: x.education == 'Bachelors', 'education'] = 'ABC'

df = DF.copy()
df['education'] = df.education.mask(df.education == 'Bachelors', 'ABC')

df = DF.copy()
df['education'] = df.education.where(df.education != 'Bachelors', 'ABC')

df = DF.copy()
df['education'] = df.education.apply(lambda x: 'ABC' if x == 'Bachelors' else x)

df = DF.copy()
df['education'] = df.education.map(lambda x: 'ABC' if x == 'Bachelors' else x)

df = DF.copy()
df.replace(to_replace={'education':'Bachelors'}, value='ABC')
```

### Homogeneity
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

uniform = np.random.rand(100)
normal = np.random.normal(size=100)
minmax_uniform = (uniform - uniform.min())/(uniform.max()-uniform.min())
minmax_normal = (normal - normal.min())/(normal.max()-normal.min())

_, axes = plt.subplots(2,1)
df = pd.DataFrame({'uniform':uniform, 'normal':normal, 'mm_uniform':minmax_uniform, 'mm_normal':minmax_normal})
df['uniform'].hist(bins=30, ax=axes[0])
df['normal'].hist(bins=30, ax=axes[1])
df.describe()
```
![image](https://user-images.githubusercontent.com/52376448/97966540-4ff11e00-1dff-11eb-9f07-e670a29ad804.png)
<br><br><br>


### Resampling
```python
import pandas as pd

time = pd.date_range('2020-01-01', periods=3, freq='D')
series = pd.Series([1,2,3])
series.index = time

series.resample('h').mean()
```
![image](https://user-images.githubusercontent.com/52376448/97406510-b0361a80-193c-11eb-9f8f-24906e2e6e1f.png)

```python
import pandas as pd

time = pd.date_range('2020-01-01', periods=3, freq='D')
series = pd.Series([1,2,3])
series.index = time

series.resample('h').mean().interpolate('linear')
```
![image](https://user-images.githubusercontent.com/52376448/97406475-a14f6800-193c-11eb-97c7-73964d7629ae.png)

```python
import FinanceDataReader as fdr

stock = fdr.DataReader('005930', start='2020-04-01')['Close']
stock.resample('D').mean().interpolate('linear')
```


<br><br><br>

### Visualization : plotly
`Installation`
```bash
$ pip install cufflinks
$ pip install chart-studio
```
```python
import chart_studio.plotly as py
import cufflinks as cf
cf.go_offline(connected=True)
cf.set_config_file(theme='pearl')
cf.getThemes()
```
- ['ggplot', 'pearl', 'solar', 'space', 'white', 'polar', 'henanigans']

```python
import numpy as np
import pandas as pd
import chart_studio.plotly as py
import cufflinks as cf
cf.go_offline(connected=True)

layout = dict()
layout['font'] = {'family': 'consolas',
                  'size': 20,
                  'color': 'blue'}
layout['title'] = "Title"

df = pd.DataFrame(np.random.rand(10, 4), columns=['A', 'B', 'C', 'D'])
df.iplot(kind='line', theme='space', layout=layout)
""" kind list """
# scatter
# bar
# box
# spread
# ratiom
# heatmap
# surface
# histogram
# bubble
# bubble3d
# scatter3d       
# scattergeo
# ohlc
# candle
# pie
# choroplet
```


