## Matplotlib
`subplot2grid`
```python
from matplotlib import pyplot as plt
plt.style.use('seaborn-notebook') # plt.style.available

gridcols = 4
table_columns = 10
quotient = table_columns//gridcols
reminder = table_columns%gridcols
layout = (quotient, gridcols) if reminder==0 else (quotient+1, gridcols)
fig = plt.figure(figsize=(25, layout[0]*5))
axes = dict()
for i in range(0, layout[0]):
    for j in range(0, layout[1]):
        idx = i*layout[1] + j
        axes[idx]= plt.subplot2grid(layout, (i, j))
```

### Installation
### Numerical Variables
```python
import pandas as pd
from ailever.dataset import UCI
from matplotlib import pyplot as plt
plt.style.use('seaborn-notebook') # plt.style.available

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['fnlwgt'] = df['fnlwgt'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)
df['education-num'] = df['education-num'].astype(float)
df = df[['age', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss', 'education-num']].copy()

gridcols = 4
table_columns = df.shape[1]

quotient = table_columns//gridcols
reminder = table_columns%gridcols
layout = (quotient, gridcols) if reminder==0 else (quotient+1, gridcols)
fig = plt.figure(figsize=(25, layout[0]*5))
axes = dict()
for i in range(0, layout[0]):
    for j in range(0, layout[1]):
        idx = i*layout[1] + j
        axes[idx]= plt.subplot2grid(layout, (i, j))

for idx, column in enumerate(numerical_columns):
    df[column].plot.hist(ax=axes[idx]) # plot.hist with edgecolor='white', plot.kde(plot.density), plot.box
plt.tight_layout()
```

### Categorical Variables
```python
import pandas as pd
from ailever.dataset import UCI
from matplotlib import pyplot as plt
plt.style.use('seaborn-notebook') # plt.style.available

df = UCI.adult(download=False)
df = df[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', '50K']].copy()

gridcols = 4
table_columns = df.shape[1]

quotient = table_columns//gridcols
reminder = table_columns%gridcols
layout = (quotient, gridcols) if reminder==0 else (quotient+1, gridcols)
fig = plt.figure(figsize=(25, layout[0]*5))
axes = dict()
for i in range(0, layout[0]):
    for j in range(0, layout[1]):
        idx = i*layout[1] + j
        axes[idx]= plt.subplot2grid(layout, (i, j))

for idx, column in enumerate(categorical_columns):
    df[column].value_counts(ascending=True).plot.bar(ax=axes[idx]) # plot.bar, plot.barh, plot.pie 
plt.tight_layout()
```

<br><br><br>

---

## Seaborn
### Installation
### Numerical Variables
```python
from ailever.dataset import UCI
import seaborn as sns
sns.set_theme(context='notebook', style='ticks') 
#sns.set_style('ticks') # darkgrid, whitegrid, dark, white, ticks
#sns.set_context("notebook") # paper, notebook, talk, poster

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['fnlwgt'] = df['fnlwgt'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)
df['education-num'] = df['education-num'].astype(float)
df_n = df[['age', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss', 'education-num']].copy()  # numerical variables
df_c = df[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', '50K']].copy() # categorical variables

g = sns.FacetGrid(df, col="workclass",  row="education") # grid by categorical variables
g.map(sns.scatterplot, "age", "hours-per-week")
```

### Categorical Variables
```python
```

<br><br><br>

---


## Plotly
### Installation
### Numerical Variables
### Categorical Variables

<br><br><br>

---


## Dash
### Installation
### Numerical Variables
### Categorical Variables

<br><br><br>

---

