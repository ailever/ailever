![image](https://user-images.githubusercontent.com/56889151/154094113-01d51f8c-04ed-4d05-8b64-1a80b372fa17.png)

### Pandas: groupby > Hierarchical Group Analysis
#### dataframe.groupby
```python
import tensorflow as tf
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

print('number of grouping:', df.groupby(['sex', 'race']).ngroups)
df['ngroup'] =               df.groupby(['sex', 'race']).ngroup() 
df
```
![image](https://user-images.githubusercontent.com/56889151/154099274-88d350fa-2170-4f68-b3bd-86c7bcb3c7da.png)


`groupby.groups` : by group index
```python
import tensorflow as tf
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

groups = df.groupby(['sex', 'race']).groups
by_group_index = pd.DataFrame(data=tf.keras.preprocessing.sequence.pad_sequences(groups.values(), padding='post'), index = pd.Index(groups.keys())).T
by_group_index.index.name = 'Index'
by_group_index
```
![image](https://user-images.githubusercontent.com/56889151/154093445-fb3946ba-96a0-46d4-851f-fe975bf78a12.png)

#### series.groupby
```python

```


### Pandas: Pivot, Crosstab > Frequency Analysis
`frequency analysis(crosstab)`
```python
import pandas as pd
from ailever.dataset import UCI

def contingency(table, prob=0.95):
    from scipy import stats
    import matplotlib.pyplot as plt

    # interpret p-value
    stat, p, dof, expected = stats.chi2_contingency(table)
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    plt.pcolor(table)
    plt.colorbar()    
    return stat, p, dof, expected
    
df = UCI.adult(download=False)
df = pd.crosstab(index=[df['education']], columns=[df['sex']], margins=True, margins_name='All', dropna=True, normalize=False) # .unstack(level=0).stack(level=1)
stat, p, dof, expected = contingency(df.values, 0.95)
```
![image](https://user-images.githubusercontent.com/56889151/151664640-a2aef54a-a48b-4a42-b6cf-dbf56973b042.png)


`frequency analysis(pivot)`
```python
import pandas as pd
from ailever.dataset import UCI

def contingency(table, prob=0.95):
    from scipy import stats
    import matplotlib.pyplot as plt

    # interpret p-value
    stat, p, dof, expected = stats.chi2_contingency(table)
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    plt.pcolor(table)
    plt.colorbar()    
    return stat, p, dof, expected

df = UCI.adult(download=False)
df = pd.pivot_table(df, index=['education'], columns='sex', values='capital-gain', aggfunc=['count']).fillna(0) # .unstack(level=0).stack(level=1)
stat, p, dof, expected = contingency(df.values, 0.95)
```
![image](https://user-images.githubusercontent.com/56889151/151664718-8ad88e6a-c945-4c97-9c44-570ccd00aa71.png)



`conditional frequency analysis(crosstab)`
```python
import pandas as pd
from ailever.dataset import UCI

def contingency(table, prob=0.95):
    from scipy import stats
    import matplotlib.pyplot as plt

    # interpret p-value
    stat, p, dof, expected = stats.chi2_contingency(table)
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    plt.pcolor(table)
    plt.colorbar()    
    return stat, p, dof, expected

df = UCI.adult(download=False)
df = pd.crosstab(index=[df['marital-status'], df['education']], columns=[df['sex']], margins=True, margins_name='All', dropna=True, normalize=False) # .unstack(level=0).stack(level=1)

#df = df.xs(key=df.index[0][0], level=df.index.names[0], axis=0)
df = df.xs(key=' Divorced', level=df.index.names[0], axis=0) 
stat, p, dof, expected = contingency(df.values[:, :-1], 0.95)
```
![image](https://user-images.githubusercontent.com/56889151/151011682-871436ed-8909-4a5b-a12e-8a631675fa92.png)

- df.columns
- df.columns.names
- df.index
- df.index.names
- df.xs(key=' Divorced', level=df.index.names[0], axis=0)
- df.xs(key=' Female', level=df.columns.names[1], axis=1)
- df.xs(key=df.index[0][0], level=df.index.names[0], axis=0)
- df.xs(key=df.columns[0][1], level=df.columns.names[1], axis=1)


`conditional frequency analysis(pivot)`
```python
import pandas as pd
from ailever.dataset import UCI

def contingency(table, prob=0.95):
    from scipy import stats
    import matplotlib.pyplot as plt

    # interpret p-value
    stat, p, dof, expected = stats.chi2_contingency(table)
    alpha = 1.0 - prob
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    plt.pcolor(table)
    plt.colorbar()    
    return stat, p, dof, expected

df = UCI.adult(download=False)
df = pd.pivot_table(df, index=['marital-status', 'education'], columns='sex', values='capital-gain', aggfunc=['count']).fillna(0) # .unstack(level=0).stack(level=1)

#df = df.xs(key=df.index[0][0], level=df.index.names[0], axis=0)
df = df.xs(key=' Divorced', level=df.index.names[0], axis=0)
stat, p, dof, expected = contingency(df.values, 0.95)
```
![image](https://user-images.githubusercontent.com/56889151/151012011-d5c61c6b-d305-47e2-8087-4840d52917a4.png)

- df.columns
- df.columns.names
- df.index
- df.index.names
- df.xs(key=' Divorced', level=df.index.names[0], axis=0)
- df.xs(key=' Female', level=df.columns.names[1], axis=1)
- df.xs(key=df.index[0][0], level=df.index.names[0], axis=0)
- df.xs(key=df.columns[0][1], level=df.columns.names[1], axis=1)


### Pandas: Describe > Percentile Analysis
`percentile analysis`
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

# .stack(level=0).to_frame().rename(columns={0:'Descriptive'})   # [row-directional flatten]
# .unstack(level=0).to_frame().rename(columns={0:'Descriptive'}) # [col-directional flatten]
df.describe(percentiles=[ 0.1*i for i in range(1, 10)], include='all').T 
```

`conditional percentile analysis`
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

categorical_variables = ['marital-status', 'education']
df.groupby(categorical_variables).describe(percentiles=[ 0.1*i for i in range(1, 10)], include='all').T # .unstack(level=0).stack(level=1)
```
![image](https://user-images.githubusercontent.com/56889151/151012160-2c044e6a-deb2-4505-be8b-87b4b3eeaf07.png)

#### Scipy: probplot > Distribution
```python
from scipy import stats
import matplotlib.pyplot as plt

def distribution_fitting(grid_order, label, data, dist, params=None):
    stats.probplot(data, dist=dist(**params), fit=True, plot=axes[f'{grid_order},1'])[1]
    axes[f'{grid_order},0'].hist(data, bins=int(len(data)/10), label='true', edgecolor='white')
    axes[f'{grid_order},0'].plot(data, int(len(data)/10)*dist.pdf(data, *dist.fit(data)), label=label, lw=0, marker='o', c='r')
    axes[f'{grid_order},0'].legend()
    axes[f'{grid_order},0'].grid(True)

layout=(8,2); fig = plt.figure(figsize=(25, 2*layout[0]))
axes = dict()
for grid_row in range(layout[0]):
    axes[f'{grid_row},0'] = plt.subplot2grid(layout, (grid_row,0))
    axes[f'{grid_row},1'] = plt.subplot2grid(layout, (grid_row,1))

    
data = stats.t.rvs(df=30, size=300)
distribution_fitting(grid_order=0, label='t', data=data, dist=stats.t, params={'df':15})
distribution_fitting(grid_order=1, label='norm', data=data, dist=stats.norm, params={})
distribution_fitting(grid_order=2, label='unirform', data=data, dist=stats.uniform, params={})
distribution_fitting(grid_order=3, label='f', data=data, dist=stats.f, params={'dfn':15, 'dfd':30})
distribution_fitting(grid_order=4, label='lognorm', data=data, dist=stats.lognorm, params={'s':1})
distribution_fitting(grid_order=5, label='beta', data=data, dist=stats.beta, params={'a':1, 'b':3})
distribution_fitting(grid_order=6, label='gamma', data=data, dist=stats.gamma, params={'a':1})
distribution_fitting(grid_order=7, label='expon', data=data, dist=stats.expon, params={})

plt.tight_layout()
```

#### Pandas: Cov, Corr > Correlation Analysis
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

display(df.cov().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'}))   # np.cov(df.T.values)
display(df.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'}))  # np.corrcoef(df.T.values)
```
![image](https://user-images.githubusercontent.com/56889151/151017428-b389a0fe-e587-4fe5-aeaf-225bd94f1355.png)



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
df.cov().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
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

#### Scipy: Interval Estimation
`Confidence Interval : Confidence`
```python
import numpy as np
from scipy import stats

#define sample data
data = np.array([12, 12, 13, 13, 15, 16, 17, 22, 23, 25, 26, 27, 28, 28, 29])

intervals = {}
#create 95% confidence interval for population mean weight
intervals['90'] = stats.t.interval(alpha=0.90, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data)) 
intervals['95'] = stats.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data)) 
intervals['99'] = stats.t.interval(alpha=0.99, df=len(data)-1, loc=np.mean(data), scale=stats.sem(data)) 

for interval in intervals.items():
    print(interval)
```
```python
import numpy as np
from scipy import stats

data = np.array([12, 12, 13, 13, 15, 16, 17, 22, 23, 25, 26, 27, 28, 28, 29])

confs = [0.90, 0.95, 0.99]
for conf in confs:
    t_stat = abs(stats.t.ppf((1 - conf)*0.5, len(data)-1))
    left_side = data.mean() - t_stat*data.std(ddof=1)/np.sqrt(len(data))
    right_side = data.mean() + t_stat*data.std(ddof=1)/np.sqrt(len(data))
    print(f'{conf}% ]', left_side, right_side)
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
