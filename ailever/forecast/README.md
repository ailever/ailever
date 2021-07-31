```python
from ailever.forecast import dashboard
dashboard()
```

```python
from ailever.forecast import TSA
tsa = TSA()
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
