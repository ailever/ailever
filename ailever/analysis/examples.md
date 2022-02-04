## Adult

### Exploratory Data Analysis

```python
from ailever.dataset import UCI
from ailever.analysis import EDA

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)
eda.frame
```
![image](https://user-images.githubusercontent.com/56889151/152403742-9ab7cd0b-f271-48fd-9fc4-779ba0cfbc1c.png)

```python
display(eda.table_definition())
display(eda.attributes_specification())
eda.frame.info()
```
![image](https://user-images.githubusercontent.com/56889151/152403838-8b4ba316-c731-46cc-8144-08883687b2bd.png)
![image](https://user-images.githubusercontent.com/56889151/152404253-fb41f0da-1d52-4f14-be3c-1446eca893af.png)
![image](https://user-images.githubusercontent.com/56889151/152404343-c35a844d-e391-4e5f-b9ec-be0eeb4c741d.png)
![image](https://user-images.githubusercontent.com/56889151/152404380-9a9cb3e8-2223-410b-a4a0-bd841c778b45.png)


```python
eda.cleaning(as_float=None, as_int=['fnlwgt', 'age', 'capital-gain', 'hours-per-week', 'capital-loss', 'education-num'], as_date=None, as_str=['education', 'native-country', 'workclass', 'occupation', 'race', 'relationship', '50K', 'marital-status', 'sex'], as_category=None, verbose=False)
display(eda.attributes_specification())
display(eda.frame.head())
display(eda.plot())
eda.univariate_frequency(view='summary').loc[lambda x: x.Rank <= 1]
```
![image](https://user-images.githubusercontent.com/56889151/152405651-0b3d52ee-3ad9-487f-ad63-786cf46eb228.png)
![image](https://user-images.githubusercontent.com/56889151/152406337-36688fb1-0acb-4d92-a738-dbc09fa9aab0.png)
![image](https://user-images.githubusercontent.com/56889151/152406400-93f060b5-7c40-436e-85d6-4c0f6c7952d1.png)
![image](https://user-images.githubusercontent.com/56889151/152409103-dcf188b3-0be1-4379-bcc2-27dfd762767a.png)


#### Independency Analysis(for categorical variables or numerical variables with only positive numbers)
```python
import pandas as pd

df = eda.frame.copy()
categorical_freq_table = pd.crosstab(columns=[df['50K']], index=[df['native-country']], margins=False, margins_name='All', dropna=True, normalize=True)
_ = hypothesis.chi2_contingency(categorical_freq_table.T)

df = eda.frame
numerical_freq_table = df.groupby(['50K']).describe(percentiles=[ 0.01*i for i in range(1, 100)])['fnlwgt'].loc[:, 'min':'max']
_ = hypothesis.chi2_contingency(numerical_freq_table)

df = eda.frame.loc[lambda x: x['capital-gain'] != 0].copy()
numerical_freq_table = df.groupby(['50K']).describe(percentiles=[ 0.01*i for i in range(1, 100)])['capital-gain'].loc[:, 'min':'max']
_ = hypothesis.chi2_contingency(numerical_freq_table)

df = eda.frame.loc[lambda x: x['capital-loss'] != 0].copy()
numerical_freq_table = df.groupby(['50K']).describe(percentiles=[ 0.01*i for i in range(1, 100)])['capital-loss'].loc[:, 'min':'max']
_ = hypothesis.chi2_contingency(numerical_freq_table)
```
![image](https://user-images.githubusercontent.com/56889151/152457958-e60599ec-f050-423a-9473-507648ed63c8.png)
![image](https://user-images.githubusercontent.com/56889151/152457897-243f8e34-dcdf-404a-98ad-e5eedc5da686.png)

### Correlation and Covariance Analysis(for numerical variables)
```python
display(eda.frame.cov().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'}))
display(eda.frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'}))
```
![image](https://user-images.githubusercontent.com/56889151/152468130-2cdea565-7159-424a-958d-9971383b6a11.png)


### Information Value
```python
eda.information_values(target_column='50K', visual_on=True) # visual_on for 'EventIVSum', 'EventIVAvg', 'QuasiBVF'
eda.iv_summary['column']                                    # eda.iv_summary['result']
```
![image](https://user-images.githubusercontent.com/56889151/152468926-5b43b40a-d668-44e9-a4e1-45a1190dd37c.png)
![image](https://user-images.githubusercontent.com/56889151/152468946-90483eff-de58-42c5-93ce-944739f1612e.png)
![image](https://user-images.githubusercontent.com/56889151/152468983-9f14b982-2af5-4004-8ed1-7db95c7d67eb.png)

### Feature importance
```python

```

### Concolusion
```python

```

