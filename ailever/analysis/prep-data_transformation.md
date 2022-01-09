
## Gaussian-like Transform
### yeo-johnson transform
```python
from ailever.dataset import SMAPI
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import PowerTransformer

frame = SMAPI.macrodata(download=False) # Multivariate Time Series Dataset
#scatter_matrix(frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})

# transform
pt = PowerTransformer(method='yeo-johnson') 
transformed_frame = pt.fit_transform(frame)
transformed_frame = pd.DataFrame(transformed_frame, columns=frame.columns)
#scatter_matrix(transformed_frame, figsize=(25,25), hist_kwds=dict(edgecolor='white'))
transformed_frame.hist(layout=(4,4), figsize=(25,25), edgecolor='white')
#transformed_frame.plot(kind='density', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.plot(kind='box', subplots=True, layout=(4,4), figsize=(25,25))
#transformed_frame.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '5pt'})
```

### box-cox transform(for positive-data)
```python

```
