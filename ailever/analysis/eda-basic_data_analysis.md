
### Pandas: Pivot
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df = pd.pivot_table(df, index=['marital-status', 'education'], columns='sex', values='capital-gain', aggfunc=['count'])
df.unstack(level=0).stack(level=1)
```

- df.columns
- df.columns.names
- df.index
- df.index.names
- df.xs(key=' Divorced', level=df.index.names[0], axis=0)
- df.xs(key=' Female', level=df.columns.names[1], axis=1)

### Pandas: Crosstab
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
pd.crosstab(index=[df['marital-status'], df['education']], columns=[df['sex']], margins=True, margins_name='All', dropna=True, normalize=False)
```

### Pandas: Describe
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

df.groupby(['marital-status', 'education']).describe(percentiles=[ 0.1*i for i in range(1, 10)], include='all').T
```


### Pandas: Visualization 
`Numerical Variables`
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
# categorical variables
df['sex'] = df['sex'].astype(str)
# numerical variables
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

df.hist(bins=30, grid=True, layout=(4,4), figsize=(25, 12), edgecolor='white')
df.plot(kind='density', subplots=True, grid=True, layout=(4,4), figsize=(25,12))
df.plot(kind='box', subplots=True, grid=True, layout=(4,4), figsize=(25,12))
df.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```

`Categorical Variables`
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
# categorical variables
df['sex'] = df['sex'].astype(str)
# numerical variables
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

df.boxplot(column='age', by='sex', grid=True, figsize=(25,5))
df.plot.scatter(y='age',  x='sex', c='capital-gain', grid=True, figsize=(25,5), colormap='viridis', colorbar=True)
#df['race'].value_counts().plot.barh(subplots=True, grid=True, figsize=(25,7))
#df['race'].value_counts().plot.pie(subplots=True, grid=True, figsize=(25,7))
```

### Pandas: One-Hot Encoding
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df = pd.concat([df, pd.get_dummies(df['sex'], prefix='sex')], axis=1)
df = pd.concat([df, pd.get_dummies(df['race'], prefix='race')], axis=1)
df
```

### Pandas: binning
`equal frequency binning`
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['hours-per-week'] = df['hours-per-week'].astype(int)

num_bin = 6
categorical_binning_frame, threshold = pd.qcut(df['hours-per-week'], q=num_bin, precision=6, duplicates='drop', retbins=True)
numerical_binning_frame = pd.qcut(df['hours-per-week'], q=num_bin, labels=threshold[1:], precision=6, duplicates='drop', retbins=False).astype(float)
```
`equal width binning`
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['hours-per-week'] = df['hours-per-week'].astype(int)

num_bin = 6
categorical_binning_frame, threshold = pd.cut(df['hours-per-week'], bins=num_bin, precision=6, retbins=True)
numerical_binning_frame = pd.cut(df['hours-per-week'], bins=num_bin, labels=threshold[1:], precision=6, retbins=False).astype(float)  
```

### Scikit-Learn: Preprocessing
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from ailever.dataset import UCI
from ailever.analysis import EDA

eda = EDA(UCI.adult(download=False), verbose=False)
df = eda.cleaning(as_int=['capital-loss', 'education-num', 'capital-gain', 'hours-per-week', 'age', 'fnlwgt'])
prep_df = pd.DataFrame(np.full_like(df, np.nan, dtype=float), columns=df.columns)

preprocessor = dict()
for name in df.columns:
    preprocessor[name] = LabelEncoder()
    prep_df[name] = preprocessor[name].fit_transform(df[name])
    #preprocessor[name].inverse_transform(prep_df[name])
prep_df
```
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from ailever.dataset import UCI
from ailever.analysis import EDA

eda = EDA(UCI.adult(download=False), verbose=False)
df = eda.cleaning(as_int=['capital-loss', 'education-num', 'capital-gain', 'hours-per-week', 'age', 'fnlwgt'])

preprocessor = dict()
preprocessor['feature'] = OrdinalEncoder()
preprocessor['target'] = LabelEncoder()

X = preprocessor['feature'].fit_transform(df.loc[:, df.columns != '50K'])
y = preprocessor['target'].fit_transform(df['50K'])
#preprocessor['feature'].inverse_transform(X)
#preprocessor['target'].inverse_transform(y)

new_columns = df.columns.to_list()
new_columns.pop(new_columns.index('50K'))
prep_df = pd.DataFrame(np.c_[X, y], columns=new_columns+['50K'])
prep_df
```
