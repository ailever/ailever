## [Data Analysis] | [pandas](https://pandas.pydata.org/pandas-docs/stable/reference/index.html) | [github](https://github.com/pandas-dev/pandas) | [MDIS](https://mdis.kostat.go.kr/index.do)

## Pandas-Basic
### DataFrame
```python
import pandas as pd
import numpy as np

df = pd.DataFrame([[38.0, 2.0, 18.0, 22.0, 21, np.nan],[19, 439, 6, 452, 226,232]],
                  index=pd.Index(['Tumour (Positive)', 'Non-Tumour (Negative)'], name='Actual Label:'),
                  columns=pd.MultiIndex.from_product([['Decision Tree', 'Regression', 'Random'],['Tumour', 'Non-Tumour']], names=['Model:', 'Predicted:']))
df
```

### Display Options
```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 30)

X, y = make_classification(n_samples=30, n_features=25, n_informative=4, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1)
df = pd.DataFrame(np.c_[X, y])
df
```


### Reduction Mapper
```python
import numpy as np
import pandas as pd

data = np.c_[np.arange(10), np.arange(10), np.arange(10), np.arange(10)]
df = pd.DataFrame(data=data, columns=list('ABCD'))

base_column = 'A'
def reduction_mapper(frame):
    # frame: raw series of actual-frame
    def function(series):
        nonlocal frame
        series = series + frame['B'] + frame['C'] + frame['D']
        if series > 5:
            instance = 1
            return instance # reduction
        else:
            instance = 2
            return instance # reduction
    return function(frame[base_column])

df.apply(reduction_mapper, axis=1)
```
```python
import numpy as np
import pandas as pd

data = np.c_[np.arange(10), np.arange(10), np.arange(10), np.arange(10)]
df = pd.DataFrame(data=data, columns=list('ABCD'))
pd.Series(data=list(map(lambda x:x[1]['A'] + x[1]['B'] + x[1]['C'] + x[1]['D'], df.iterrows())))
```


### Boolean Indexer
```python
import numpy as np
import pandas as pd
from ailever.analysis import EDA
from ailever.dataset import UCI

DF = EDA(UCI.adult(download=False), verbose=False).cleaning()

df = DF.copy()
df['_'] = np.nan
df.loc[lambda x: (x['education'] == 'Bachelors') & (x['native-country'] == 'United-States'), '_'] = True
df.loc[lambda x: ~((x['education'] == 'Bachelors') & (x['native-country'] == 'United-States')), '_'] = False
boolean_indexer = pd.Index(df['_'].astype(bool))
boolean_indexer

df = DF.copy()
boolean_indexer = df.education.mask((df['education'] == 'Bachelors') & (df['native-country'] == 'United-States'), '_MARKER_')
boolean_indexer = pd.Index(boolean_indexer.where(boolean_indexer == '_MARKER_', False).astype(bool))
boolean_indexer
```

### Conditional Replacement
```python
import pandas as pd
from ailever.analysis import EDA
from ailever.dataset import UCI

DF = EDA(UCI.adult(download=False), verbose=False).cleaning()

df = DF.copy()
df.loc[lambda x: x.education == 'Bachelors', 'education'] = 'ABC'

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


---


## Pandas-Advanced
### Visualization
- https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html

`built-in summary`
```python
.style.background_gradient(cmap=sns.light_palette("green", as_cmap=True))
.style.highlight_null(null_color='yellow')
.style.highlight_min(axis=0, color='red')
.style.highlight_max(axis=0, color='yellow')
```

`style.highlight_null`
```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 30)

X, y = make_classification(n_samples=30, n_features=25, n_informative=4, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1)
df = pd.DataFrame(np.c_[X, y]).applymap(lambda x: np.nan if x > 2.5 or x < -2.5 else x)
df.style.highlight_null(null_color='yellow')
```

`.style.highlight_max`, `.style.highlight_min`
```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 30)

X, y = make_classification(n_samples=30, n_features=25, n_informative=4, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1)
df = pd.DataFrame(np.c_[X, y]).applymap(lambda x: np.nan if x > 2.5 or x < -2.5 else x)
df.style.highlight_min(axis=0, color='red')
df.style.highlight_max(axis=0, color='yellow')
```

`.style.background_gradient`
```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification, make_regression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 30)
cm = sns.light_palette("green", as_cmap=True)

X, y = make_classification(n_samples=30, n_features=25, n_informative=4, n_redundant=0, n_repeated=0, n_classes=3, n_clusters_per_class=1)
df = pd.DataFrame(np.c_[X, y]).applymap(lambda x: np.nan if x > 2.5 or x < -2.5 else x)
df.style.background_gradient(cmap=cm)
```

---


## Analysis Utils
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


