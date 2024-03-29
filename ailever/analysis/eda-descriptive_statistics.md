## Table
`Uni Table Example`
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df.columns.name = 'adult'
df.index.name = 'index'
df
```
![image](https://user-images.githubusercontent.com/56889151/154117354-10e92a70-d061-4080-86ac-7f22a35209c8.png)

`Multi Table Example`
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df = pd.pivot_table(df, index=['race', 'education'], columns=['sex'], values='capital-gain', aggfunc=['sum']).fillna(0) # .unstack(level=0).stack(level=1)
df.columns.names = ['AGGREGATION', 'SEX']
df.index.names = ['RACE', 'EDUCATION', 'capital-gain']

df.reset_index('capital-gain').drop('capital-gain', level=0, axis=1)
```
![image](https://user-images.githubusercontent.com/56889151/154121832-638273b3-317a-4054-8047-04d022f3d189.png)


---


### Pandas: groupby > Hierarchical Group Analysis
#### aggregation
```python
import pandas as pd
from ailever.dataset import UCI

def agg(frame):
    def df_agg_unit(frame, column):
        agg_unit = frame.groupby(column)[[column]].count().rename(columns={column:'cnt'}).sort_values('cnt', ascending=False)
        agg_unit['ratio'] = agg_unit['cnt'].apply(lambda x: x/agg_unit['cnt'].sum())
        agg_unit['cumulative'] = agg_unit['ratio'].cumsum()
        agg_unit['rank'] = agg_unit['cnt'].rank(ascending=False)
        agg_unit.index = pd.MultiIndex.from_product([[agg_unit.index.name], agg_unit.index.tolist()])    
        return agg_unit

    agg_table = pd.concat(list(map(lambda column: df_agg_unit(frame, column=column), frame.columns)), axis=0)
    agg_table.index.names = ['column', 'instance']
    return agg_table

frame = UCI.adult(download=False)
agg_table = agg(frame)
agg_table
```

#### dataframe.groupby
`groupby.[/head/tail/sample/nth/rank]`
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

df.groupby(['sex', 'race']).head(n=5)
df.groupby(['sex', 'race']).tail(n=5)
df.groupby(['sex', 'race']).sample(n=1)
df.groupby(['sex', 'race']).sample(frac=0.01)
df.groupby(['sex', 'race']).nth(n=5)
df.groupby(['sex', 'race']).rank(method='first', axis=0)
df.groupby(['sex', 'race']).nunique()
```

`groupby.apply(lambda)`
```python
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

df['ngroup'] = df.groupby(['sex', 'race']).ngroup()
df.groupby(['sex', 'race'])['ngroup'].apply(lambda x: print(x)) # x: series by ngroup('sex', 'race')
df.groupby(['sex', 'race']).apply(lambda x: print(x)) # x: series by ngroup('sex', 'race')
```


`groupby.ngroup()`
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
df = df.sort_values(['sex', 'race'])
df['ngroup'] = df.groupby(['sex', 'race']).ngroup()
df['index'] = df.groupby(['sex', 'race']).rank(method='first', axis=0)['ngroup']
df.set_index(['sex', 'race', 'index'], verify_integrity=True)
```
![image](https://user-images.githubusercontent.com/56889151/154111226-c7a3d0cf-0d99-4c96-a458-d278992b8e7a.png)

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
by_group_index = pd.DataFrame(data=tf.keras.preprocessing.sequence.pad_sequences(groups.values(), value=-1, padding='post'), index = pd.Index(groups.keys())).T
by_group_index.index.name = 'Index'
by_group_index
```
![image](https://user-images.githubusercontent.com/56889151/154114857-3d27695c-3dea-457a-9046-0906bd3ec855.png)



#### series.groupby
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

df.groupby(['sex', 'race'])['workclass'].unique()
df.groupby(['sex', 'race'])['education'].unique()
df.groupby(['sex', 'race'])['education-num'].unique()
df.groupby(['sex', 'race'])['marital-status'].unique()
df.groupby(['sex', 'race'])['occupation'].unique()
df.groupby(['sex', 'race'])['relationship'].unique()
df.groupby(['sex', 'race'])['native-country'].unique()
df.groupby(['sex', 'race'])['50K'].unique()
```


#### Pandas: unstack, stack, swaplevel > Conditional Group Analysis
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df = pd.pivot_table(df, index=['race', 'education'], columns=['sex'], values='capital-gain', aggfunc=['sum']).fillna(0) # .unstack(level=0).stack(level=1)
df.columns.names = ['AGGREGATION', 'SEX']
df.index.names = ['RACE', 'EDUCATION', 'capital-gain']
df = df.reset_index('capital-gain').drop('capital-gain', level=0, axis=1)
df
```
![image](https://user-images.githubusercontent.com/56889151/154126154-ba414497-bbd7-45a2-bc55-e7bb5e4b30f3.png)

`unstack`: expand column (direction from index to column)
```python
df.unstack(level=1)
```
![image](https://user-images.githubusercontent.com/56889151/154124858-fdefa182-f21f-4f0c-8550-214274b810e4.png)

`stack`: contraction column (direction from column to index)
```python
df.unstack(level=1).stack(level=2)
```
![image](https://user-images.githubusercontent.com/56889151/154126159-74ce59d2-98d1-404e-b8da-1888bd2fc99e.png)

`swaplevel`
```python
df.swaplevel(i=0, j=1, axis=0).sort_index(level=0)
```
![image](https://user-images.githubusercontent.com/56889151/154127138-edf8dbf0-9a72-411d-babd-d3f59349a158.png)


---

### Pandas: Pivot, Crosstab, Groupby > Frequency Analysis
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

`frequency analysis by group`
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

print('number of grouping:', df.groupby(['sex', 'race', 'relationship']).ngroups)
df = df.sort_values(['sex', 'race', 'relationship'])
df['ngroup'] = df.groupby(['sex', 'race', 'relationship']).ngroup()
df['index'] = df.groupby(['sex', 'race', 'relationship']).rank(method='first', axis=0)['ngroup']
df = df.set_index(['sex', 'race', 'relationship', 'index'], verify_integrity=True)

high_level_classification = df.groupby(['sex']).agg('count')[['ngroup']].reset_index()
middle_level_classification = df.groupby(['sex', 'race']).agg('count')[['ngroup']].reset_index()
low_level_classification = df.groupby(['sex', 'race', 'relationship']).agg('count')[['ngroup']].reset_index()

frequency_by_group = low_level_classification.merge(middle_level_classification, how='inner', on='race', suffixes=['', '_']).rename(columns={'ngroup':'LowLevelCNT', 'ngroup_':'MiddleLevelCNT'})[['sex', 'race', 'MiddleLevelCNT', 'relationship', 'LowLevelCNT']].merge(high_level_classification, how='inner', on='sex', suffixes=['', '_']).rename(columns={'ngroup':'HighLevelCNT'})[['sex', 'HighLevelCNT', 'race', 'MiddleLevelCNT', 'relationship', 'LowLevelCNT']]
frequency_by_group
```
![image](https://user-images.githubusercontent.com/56889151/154461802-a0b25902-d118-4f40-beac-ab2d48a5320d.png)



```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['fnlwgt'] = df['fnlwgt'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)
df['workclass'] = df['workclass'].astype('category')
df['education'] = df['education'].astype('category')
df['marital-status'] = df['marital-status'].astype('category')
df['occupation'] = df['occupation'].astype('category')
df['relationship'] = df['relationship'].astype('category')
df['race'] = df['race'].astype('category')
df['sex'] = df['sex'].astype('category')
df['native-country'] = df['native-country'].astype('category')
df['50K'] = df['50K'].astype('category')

categorical_variables = ['race', 'education']
df.groupby(categorical_variables).describe(include='category').T.unstack(level=1).stack(level=0).stack(level=0).style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```
![image](https://user-images.githubusercontent.com/56889151/154468705-9e262fb7-13e1-4ad7-9158-15f33baed3c9.png)



---

### Pandas: Groupby > Percentile Analysis
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
df.describe(percentiles=[ 0.1*i for i in range(1, 10)], include='all').T.style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'}) 
```

`conditional percentile analysis(1): bottom-up interface`
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

categorical_variables = ['race', 'education']
df.groupby(categorical_variables).describe(percentiles=[ 0.1*i for i in range(1, 10)]).T.unstack(level=1).stack(level=0).stack(level=0).style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```
![image](https://user-images.githubusercontent.com/56889151/154134340-58a9fade-d043-4527-a5b9-3fb0ab409984.png)


`conditional percentile analysis(2): top-down interface`
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)

df['ngroup'] =               df.groupby(['sex', 'race']).ngroup() 
df = df.sort_values(['sex', 'race']).reset_index(drop=True)
df['index'] = df.groupby(['sex', 'race']).rank(method='first', axis=0)['ngroup']
df = df.set_index(['sex', 'race', 'index'], verify_integrity=True)
df.unstack(level=1).xs(key=' Female', level=0, axis=0).describe().T.style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```
![image](https://user-images.githubusercontent.com/56889151/154133453-0fd014de-852d-4f7e-a1c7-d13081ae7808.png)



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


---

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


---

### Preprocessing
#### Scikit-Learn: Preprocessing
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

#### Pandas: One-Hot Encoding
```python
import pandas as pd
from ailever.dataset import UCI

df = UCI.adult(download=False)
df = pd.concat([df, pd.get_dummies(df['sex'], prefix='sex')], axis=1)
df = pd.concat([df, pd.get_dummies(df['race'], prefix='race')], axis=1)
df
```

#### Pandas: crosstab
`Term Frequency-Inverse Document Frequency`
```python
import pandas as pd
import numpy as np
from ailever.dataset import UCI

df = UCI.adult(download=False)

n = 100
tf = pd.crosstab(df['education'], [df['occupation']], margins=True).sort_values('All', ascending=False).iloc[1:]
idf = tf.applymap(lambda x: 1 if x > 0 else 0)
tfidf = tf * idf.sum().apply(lambda x: np.log(n/(x+1)))
tfidf
```


#### Pandas: binning
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

##### Scipy: Interval Estimation
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

---

### Summary

```python
from ailever.dataset import UCI

# DataLoad
df = UCI.adult(download=False)

# CNT LEVEL3: race
summary_table = df.groupby(['native-country', 'sex', 'race'])[['race']].count().rename(columns={'race':'race_cnt'})

# CNT LEVEL2: sex
summary_table['sex_cnt'] = summary_table.groupby(['native-country', 'sex'])['race_cnt'].apply(lambda x: x.sum())

# CNT LEVEL1: native-country
country_cnt = summary_table.groupby(['native-country'])['sex_cnt'].apply(lambda x: x.sum())
country_cnt.index = pd.MultiIndex.from_frame(country_cnt.index.to_frame())
summary_table['country_cnt'] = country_cnt

# RATIOs
summary_table['country_ratio'] = summary_table['country_cnt'] / summary_table['race_cnt'].sum() # RATIO LEVEL1: native-country
summary_table['sex_ratio'] = summary_table['sex_cnt'] / summary_table['country_cnt'].sum()      # RATIO LEVEL2: sex
summary_table['race_ratio'] = summary_table['race_cnt'] / summary_table['sex_cnt'].sum()        # RATIO LEVEL3: race
summary_table = summary_table.reset_index()[['native-country', 'country_cnt', 'country_ratio', 'sex', 'sex_cnt', 'sex_ratio', 'race', 'race_cnt', 'race_ratio']]
summary_table
```
![image](https://user-images.githubusercontent.com/56889151/156551838-0030d21b-9334-4d71-884d-037b63d6f8e3.png)

