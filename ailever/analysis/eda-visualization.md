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

`sns.histplot`
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
g.map_dataframe(sns.histplot, x="age", binwidth=2, binrange=(0, 60), kde=False) # x: numerical variable / sns.histplot
#g.map(sns.histplot, "age", binwidth=2, binrange=(0, 60), kde=False)
g.add_legend()
g.tight_layout()
```

`sns.scatterplot`
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
g.map_dataframe(sns.scatterplot, x="age", y="hours-per-week", hue='sex') # x: numerical variable, y: numerical variable, hue: categorical variable / sns.scatterplot, sns.lineplot
#g.map(sns.scatterplot, "age", "hours-per-week")
g.add_legend()
g.tight_layout()
```

`sns.boxplot`
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
g.map_dataframe(sns.boxplot, x="age") # x: numerical variable / sns.boxplot(outlier), sns.violinplot(variance), sns.boxenplot(variance), sns.stripplot(distribution), sns.pointplot(mean, variance), sns.barplot(mean)
g.add_legend()
g.tight_layout()
```



### Categorical Variables
`sns.countplot`
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
g.map_dataframe(sns.countplot, x="marital-status", hue='occupation') # x(vertical while y means horizental): categorical variable, hue: categorical variable
g.add_legend()
g.tight_layout()
```

`sns.boxplot`
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
g.map_dataframe(sns.boxplot, x="age", y="hours-per-week", hue="sex") # x: numerical variable, y: numerical variable, hue: categorical variable
g.add_legend()
g.tight_layout()
```

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
g.map_dataframe(sns.boxplot, x="marital-status", y="hours-per-week", hue="sex", orient="v") # x: categorical variable, y: numerical variable, hue: categorical variable / sns.boxplot(outlier), sns.violinplot(variance), sns.boxenplot(variance), sns.stripplot(distribution), sns.barplot(mean)
g.add_legend()
g.tight_layout()
```

<br><br><br>

---


## Plotly
### Installation
### Numerical Variables
`px.scatter`
```python
import plotly.express as px
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['fnlwgt'] = df['fnlwgt'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)
df['education-num'] = df['education-num'].astype(float)
df_n = df[['age', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss', 'education-num']].copy()  # numerical variables
df_c = df[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', '50K']].copy() # categorical variables

fig = px.scatter(df, x="capital-loss", y="capital-gain", size='age', color="relationship", hover_name="native-country", log_x=True, size_max=60) # x: numerical variable, y: numerical variable, size: numerical variable, color: categorical variable, hover_name: categorical variable
fig.show()
```

### Categorical Variables
`px.bar`
```
import plotly.express as px
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['fnlwgt'] = df['fnlwgt'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)
df['education-num'] = df['education-num'].astype(float)
df_n = df[['age', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss', 'education-num']].copy()  # numerical variables
df_c = df[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', '50K']].copy() # categorical variables

fig = px.bar(df, x="workclass", y="capital-gain", color="education", title="TITLE") # x: categorical variable, y: numerical variable, color: categorical variable
fig.show()
```

`px.sunburst`
```python
import plotly.express as px
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['fnlwgt'] = df['fnlwgt'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)
df['education-num'] = df['education-num'].astype(float)
df_n = df[['age', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss', 'education-num']].copy()  # numerical variables
df_c = df[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', '50K']].copy() # categorical variables

fig = px.sunburst(df, path=['education', 'relationship', 'race', 'sex'], values='capital-gain')
fig.show()
```

`px.treemap`
```python
import plotly.express as px
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['fnlwgt'] = df['fnlwgt'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)
df['education-num'] = df['education-num'].astype(float)
df_n = df[['age', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss', 'education-num']].copy()  # numerical variables
df_c = df[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', '50K']].copy() # categorical variables

fig = px.treemap(df, path=[px.Constant("all"), 'education', 'relationship', 'race', 'sex'], values='capital-gain')
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()
```

`px.icicle`
```python
import plotly.express as px
from ailever.dataset import UCI

df = UCI.adult(download=False)
df['age'] = df['age'].astype(int)
df['hours-per-week'] = df['hours-per-week'].astype(int)
df['fnlwgt'] = df['fnlwgt'].astype(int)
df['capital-gain'] = df['capital-gain'].astype(float)
df['capital-loss'] = df['capital-loss'].astype(float)
df['education-num'] = df['education-num'].astype(float)
df_n = df[['age', 'hours-per-week', 'fnlwgt', 'capital-gain', 'capital-loss', 'education-num']].copy()  # numerical variables
df_c = df[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', '50K']].copy() # categorical variables

fig = px.icicle(df, path=[px.Constant("all"), 'education', 'relationship', 'race', 'sex'], values='capital-gain')
fig.update_traces(root_color="lightgrey")
fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()
```

<br><br><br>

---


## Dash
### Installation
### Numerical Variables
### Categorical Variables

<br><br><br>

---


## Excel
### Installation
### Numerical Variables
### Categorical Variables

<br><br><br>

---

