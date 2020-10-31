## [Data Analysis] | [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/index.html) | [github](https://github.com/pandas-dev/pandas)


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


