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
