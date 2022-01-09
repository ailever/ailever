## Tranformation for Numerical variables
### Polynomial Feature Tranformation
```python
```

### Quantile Tranformation
#### Normal Quantile Tranformation
```python
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import QuantileTransformer
from ailever.dataset import SMAPI

# [origin]
frame = SMAPI.macrodata(download=False) # Multivariate Time Series Dataset
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [transform]
transformer = QuantileTransformer(n_quantiles=100, output_distribution='normal') 
transformed_frame = transformer.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

recovered_frame = transformer.inverse_transform(transformed_frame)
recovered_frame = pd.DataFrame(recovered_frame, columns=frame.columns)
#scatter_matrix(recovered_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
recovered_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#recovered_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

```

#### Uniform Quantile Tranformation
```python
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import QuantileTransformer
from ailever.dataset import SMAPI

# [origin]
frame = SMAPI.macrodata(download=False) # Multivariate Time Series Dataset
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [transform]
transformer = QuantileTransformer(n_quantiles=100, output_distribution='uniform') 
transformed_frame = transformer.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

recovered_frame = transformer.inverse_transform(transformed_frame)
recovered_frame = pd.DataFrame(recovered_frame, columns=frame.columns)
#scatter_matrix(recovered_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
recovered_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#recovered_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```


### Scaling Transformation
#### Minmax Scaling
```python
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import MinMaxScaler
from ailever.dataset import SMAPI

# [origin]
frame = SMAPI.macrodata(download=False) # Multivariate Time Series Dataset
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [transform]
transformer = MinMaxScaler() 
transformed_frame = transformer.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

recovered_frame = transformer.inverse_transform(transformed_frame)
recovered_frame = pd.DataFrame(recovered_frame, columns=frame.columns)
#scatter_matrix(recovered_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
recovered_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#recovered_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```

#### Standard Scaling
```python
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from ailever.dataset import SMAPI

# [origin]
frame = SMAPI.macrodata(download=False) # Multivariate Time Series Dataset
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [transform]
transformer = StandardScaler() 
transformed_frame = transformer.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

recovered_frame = transformer.inverse_transform(transformed_frame)
recovered_frame = pd.DataFrame(recovered_frame, columns=frame.columns)
#scatter_matrix(recovered_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
recovered_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#recovered_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```

#### Robust Scaling
```python
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import RobustScaler
from ailever.dataset import SMAPI

# [origin]
frame = SMAPI.macrodata(download=False) # Multivariate Time Series Dataset
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [transform]
transformer = RobustScaler() 
transformed_frame = transformer.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

recovered_frame = transformer.inverse_transform(transformed_frame)
recovered_frame = pd.DataFrame(recovered_frame, columns=frame.columns)
#scatter_matrix(recovered_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
recovered_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#recovered_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```


### Gaussian-like Tranformation
#### yeo-johnson Tranformation
```python
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import PowerTransformer
from ailever.dataset import SMAPI

# [origin]
frame = SMAPI.macrodata(download=False) # Multivariate Time Series Dataset
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [transform]
transformer = PowerTransformer(method='yeo-johnson') 
transformed_frame = transformer.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [inverse tranform]
recovered_frame = transformer.inverse_transform(transformed_frame)
recovered_frame = pd.DataFrame(recovered_frame, columns=frame.columns)
#scatter_matrix(recovered_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
recovered_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#recovered_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```

#### box-cox Tranformation(for positive-data)
```python
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import PowerTransformer
from ailever.dataset import SMAPI

# [origin]
frame = SMAPI.macrodata(download=False).drop(['infl', 'realint'], axis=1) # Multivariate Time Series Dataset
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [transform]
transformer = PowerTransformer(method='box-cox') 
transformed_frame = transformer.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [inverse tranform]
recovered_frame = transformer.inverse_transform(transformed_frame)
recovered_frame = pd.DataFrame(recovered_frame, columns=frame.columns)
#scatter_matrix(recovered_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
recovered_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#recovered_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```


---

## Transformation for Categorical variables
### Numerical Encoding
```python
from ailever.dataset import UCI 
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder 

frame = UCI.breast_cancer(download=False) 

encoder1 = LabelEncoder()
encoder2 = OrdinalEncoder()
encoder3 = OneHotEncoder()

numerical_column1 = encoder1.fit_transform(frame['menopause'])
numerical_column2 = encoder2.fit_transform(frame[['menopause']])
numerical_column3 = encoder3.fit_transform(frame[['menopause']])

origin_column1 = encoder1.inverse_transform(numerical_column1)
origin_column2 = encoder2.inverse_transform(numerical_column2)
origin_column3 = encoder3.inverse_transform(numerical_column3)
```



### Discretization Transformation
#### Uniform Discretization Transformation
```python
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import KBinsDiscretizer
from ailever.dataset import SMAPI

# [origin]
frame = SMAPI.macrodata(download=False) # Multivariate Time Series Dataset
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [transform]
transformer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform') 
transformed_frame = transformer.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

recovered_frame = transformer.inverse_transform(transformed_frame)
recovered_frame = pd.DataFrame(recovered_frame, columns=frame.columns)
#scatter_matrix(recovered_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
recovered_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#recovered_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```

#### k-means Discretization Transformation
```python
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import KBinsDiscretizer
from ailever.dataset import SMAPI

# [origin]
frame = SMAPI.macrodata(download=False) # Multivariate Time Series Dataset
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [transform]
transformer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans') 
transformed_frame = transformer.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

recovered_frame = transformer.inverse_transform(transformed_frame)
recovered_frame = pd.DataFrame(recovered_frame, columns=frame.columns)
#scatter_matrix(recovered_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
recovered_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#recovered_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```

#### Qunatile Discretization Transformation
```python
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import KBinsDiscretizer
from ailever.dataset import SMAPI

# [origin]
frame = SMAPI.macrodata(download=False) # Multivariate Time Series Dataset
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# [transform]
transformer = KBinsDiscretizer(n_bins=12, encode='ordinal', strategy='quantile') 
transformed_frame = transformer.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

recovered_frame = transformer.inverse_transform(transformed_frame)
recovered_frame = pd.DataFrame(recovered_frame, columns=frame.columns)
#scatter_matrix(recovered_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
recovered_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#recovered_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#recovered_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```


