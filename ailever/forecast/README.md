# Forecast Package

```python
from ailever.forecast import dashboard
dashboard()
```

```python
from ailever.forecast import sarima
trendAR=[]; trendMA=[]
seasonAR=[]; seasonMA=[]
process = sarima.Process((1,1,2), (2,0,1,4), trendAR=trendAR, trendMA=trendMA, seasonAR=seasonAR, seasonMA=seasonMA, n_samples=300)
process.final_coeffs
process.TS_Yt
process.samples
```


```python
from ailever.forecast import TSA
from ailever.investment import prllz_loader

datacore = prllz_loader(baskets=['ARE', 'BXP', 'O'])
datacore.ndarray
datacore.pdframe

tsa = TSA(datacore.pdframe, 'ARE')
tsa.Correlation(datacore.pdframe, column_sequence=['ARE', 'BXP', 'O'])
tsa.STL()
```


```python
from ailever.forecast import TSA
from ailever.investment import Loader

loader = Loader()
Df = loader.ohlcv_loader(baskets=['ARE', 'O', 'BXP'])
profit = Df.dict['ARE']

tsa = TSA(frame=profit, target_column='close')
tsa.SARIMAX()
tsa.ExponentialSmoothing()
tsa.ETS()
tsa.Prophet()
tsa.VAR()
tsa.VECM()
tsa.GARCH()
```


```python
from ailever.forecast import TSAM

tsam = TSAM(frame)
tsam.SARIMAX(time, model_params)
tsam.ExponentialSmoothing(time, model_params)
tsam.ETS(time, model_params)
tsam.Prophet(time, model_params)
tsam.XGBoost(time, model_params)
tsam.MLP(time, model_params)
tsam.CNN(time, model_params)
tsam.LSTM(time, model_params)
tsam.GRU(time, model_params)
tsam.BERT(time, model_params)
tsam.GPT(time, model_params)
```


## Forecasting model reviews
### SARIMAX
```python
from ailever.dataset import SMAPI
import statsmodels.tsa.api as smt
import pandas as pd

frame = SMAPI.co2(download=False).dropna()
frame = frame.asfreq('M').fillna(method='bfill').fillna(method='ffill')

trend = [None, 'c', 't', 'ct']
model = smt.SARIMAX(frame['co2'], order=(1,0,1), seasonal_order=(1,1,2,7), trend=trend[0], freq='M', simple_differencing=False)
model = model.fit(disp=False)

alpha = 0.05
steps = 10
prediction = pd.concat([model.predict(),
                        model.get_prediction().conf_int(alpha=alpha)], axis=1)
forecast = pd.concat([model.forecast(steps=steps),
                      model.get_forecast(steps=steps).conf_int(alpha=alpha)], axis=1)

prediction_table = prediction.append(forecast)
prediction_table
```
- model.states.filtered
- model.states.filtered_cov
- model.states.predicted
- model.states.predicted_cov
- model.states.smoothed
- model.states.smoothed_cov

- model.summary()
- model.params
- model.arparams
- model.maparams
- model.seasonalarparams
- model.seasonalmaparams
- model.pvalues
- model.tvalues
- model.zvalues
- model.mse
- model.mae

- model.plot_diagnostics(figsize=(25,7))
- model.predict()
- model.impulse_responses(steps=10, impulse=0, orthogonalized=False)

### VAR
```python
```

### Prophet
```python
```
