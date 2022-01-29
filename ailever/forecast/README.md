# Forecast Package
- [Practical Econometrics and Data Science](http://web.vu.lt/mif/a.buteikis/wp-content/uploads/PE_Book/index.html)
- [Forecasting: Principles and Practice](https://otexts.com/fpp3/)
- [Quantitative Economics with Python](https://python.quantecon.org/intro.html)
- [statsmodels](https://www.statsmodels.org/stable/examples/index.html)
- [Examples](https://github.com/ailever/ailever/blob/master/ailever/forecast/examples.md)


```python
from ailever.forecast import dashboard
dashboard()
```

## REVIEW
### Time Offset
```python
import pandas as pd
from ailever.dataset import SMAPI

df = SMAPI.macrodata(download=False)
df.index = pd.date_range(start='1959-01-01', periods=df.shape[0], freq='Q')
df = df.asfreq('Q').fillna(method='ffill').fillna(method='bfill')
df
```

```python
import FinanceDataReader as fdr

df = fdr.DataReader('ARE')
df = df.asfreq('B').fillna(method='ffill').fillna(method='bfill')
df
```

### Datetime Index
```python
import pandas as pd

pd.DatetimeIndex(['2010-01-01'], freq='B').shift(5)[0].strftime('%Y-%m-%d')
```
```python
import pandas as pd

df = pd.DataFrame()
df['date'] = pd.date_range(start='2000-01-01', periods=10000, freq='d')
df.set_index('date', inplace=True)
df = df.asfreq('d').fillna(method='ffill').fillna(method='bfill')

df['year'] = df.index.year
df['quarterofyear'] = df.index.quarter
df['monthofyear'] = df.index.month
df['weekofyear'] = df.index.isocalendar().week # week of year
df['dayofyear'] = df.index.dayofyear
df['dayofmonth'] = df.index.day
df['dayofweek'] = df.index.dayofweek
df
```



## TSA


```python
from ailever.forecast import sarima
trendAR=[]; trendMA=[]
seasonAR=[]; seasonMA=[]
process = sarima.Process((1,1,2), (2,0,1,4), trendAR=trendAR, trendMA=trendMA, seasonAR=seasonAR, seasonMA=seasonMA, n_samples=300, verbose=False)
```
```python
process.final_coeffs
process.TS_Yt
process.samples
process.experimental_acf
process.experimental_pacf
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

```python
from ailever.dataset import SKAPI
from ailever.forecast import LightMLOps

dataset = SKAPI.boston(download=False)
lmlops = LightMLOps(dataset=dataset, target='target')
lmlops.trigger(fine_tuning=False)
lmlops.feature_store()
lmlops.model_registry()
lmlops.analysis_report_repository()
```

<br><br><br>

---




## REVIEW : Time Series Analysis
### [Forecasting Model] SARIMAX
```python
from ailever.dataset import SMAPI
import statsmodels.tsa.api as smt
import pandas as pd

frame = SMAPI.co2(download=False).dropna()
frame = frame.asfreq('M').fillna(method='bfill').fillna(method='ffill')

trend = [None, 'c', 't', 'ct'] # 'c' : intercept, 't' : 'drift', 'ct' : 'intercept'+'drift'
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
- model.impulse_responses(steps=10, impulse=0, orthogonalized=False)

### [Forecasting Model] ETS
```python
```

### [Forecasting Model] VAR
```python
# https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/
# https://www.machinelearningplus.com/time-series/vector-autoregression-examples-python/

from ailever.dataset import SMAPI
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

macrodata = SMAPI.macrodata(download=False)
macrodata['year'] = macrodata['year'].astype(int).astype(str)
macrodata['md'] = macrodata.quarter.apply(lambda x: '-01-01' if x==1 else '-04-01' if x==2 else '-07-01' if x==3 else '-10-01')
macrodata['date'] = pd.to_datetime(macrodata.assign(date=lambda x: x.year + x.md)['date'])
macrodata = macrodata.set_index('date').asfreq('QS').drop('md', axis=1)
time_series = macrodata[['realgdp', 'realcons']].diff().dropna()

maxlags = 3
for idx, column in enumerate(time_series.columns):
    print(f'[Notation] y{idx+1}: {column}')
model = sm.tsa.VAR(time_series.values).fit(maxlags=maxlags)
model.irf(10).plot(figsize=(25,7))
plt.tight_layout()
#model.summary()
#model.params
#model.coefs
#model.coefs_exog # model.intercept
#model.sigma_u
#model.pvalues
#model.tvalues

# Cointegration Test
print('---------'*10)
samples = time_series.values
statistic, pvalue, _ = sm.tsa.coint(samples[:, 0], samples[:, 1])
if pvalue < 0.05:
    print(f"[Cointegration Test] : Two time series have cointegration relation(p-value : {pvalue}).")
else:
    print(f"[Cointegration Test] : Two time series don't have cointegration relation(p-value : {pvalue}).")


# Granger Causality Test
print('---------'*10)
samples1 = np.c_[samples[:,0], samples[:,1]]
samples2 = np.c_[samples[:,1], samples[:,0]]
print('[Granger Causality Test] : y2 -> y1')
sm.tsa.stattools.grangercausalitytests(samples1, maxlag=3*maxlags, verbose=True)
print('\n[Granger Causality Test] : y1 -> y2')
sm.tsa.stattools.grangercausalitytests(samples2, maxlag=3*maxlags, verbose=True)

# Forecast
print('---------'*10)
steps = 20
forecasting_values = model.forecast(y=time_series.values[-model.k_ar:], steps=steps)
#prediction_values_ = np.r_[macrodata[['realgdp', 'realcons']].iloc[0].values[np.newaxis, :], time_series.values, forecasting_values].cumsum(axis=0)
#prediction_table_ = pd.DataFrame(data=prediction_values_, index=pd.date_range(time_series.index[0], periods=macrodata.shape[0]+steps, freq='QS'), columns=time_series.columns)
prediction_values = np.r_[macrodata[['realgdp', 'realcons']].iloc[:maxlags+1].values, model.fittedvalues, forecasting_values].cumsum(axis=0)
prediction_table = pd.DataFrame(data=prediction_values, index=pd.date_range(time_series.index[0], periods=macrodata.shape[0]+steps, freq='QS'), columns=time_series.columns)
prediction_table
```

### [Forecasting Model] Prophet
```python
```



## StockProphet
### evaluate
`by lag_shift`
```python
import pandas as pd
from ailever.forecast import StockProphet
pd.set_option('display.max_columns', None)

prophet = StockProphet(code='005390', lag_shift=5, sequence_length=10)
for i in range(6, 30):
    prophet.evaluate(model_name='GradientBoostingClassifier', trainstartdate='2015-03-01', teststartdate='2019-10-01', code=None, lag_shift=i, comment=None, visual_on=False)

#prophet.dataset
#prophet.model
prophet.evaluation
```

`by code and lag_shift`
```python
import pandas as pd
pd.set_option('display.max_columns', None)

import FinanceDataReader as fdr
marcap_table = fdr.StockListing('KRX-MARCAP')
marcap_table.iloc[:10] 

from ailever.forecast import StockProphet
prophet = StockProphet(code='ARE', lag_shift=5, sequence_length=10)
for i in range(6, 30):
    prophet.evaluate(model_name='GradientBoostingClassifier', trainstartdate='2015-03-01', teststartdate='2019-10-01', code=None, lag_shift=i, comment=None, visual_on=False)
for i in range(5, 30):
    prophet.evaluate(model_name='GradientBoostingClassifier', trainstartdate='2015-03-01', teststartdate='2019-10-01', code='BXP', lag_shift=i, comment=None, visual_on=False)
for i in range(5, 30):
    prophet.evaluate(model_name='GradientBoostingClassifier', trainstartdate='2015-03-01', teststartdate='2019-10-01', code='O', lag_shift=i, comment=None, visual_on=False)

#prophet.dataset
#prophet.model
prophet.evaluation
```

`by trainstartdate and teststartdate`
```python
import numpy as np
import pandas as pd
from ailever.forecast import StockProphet
pd.set_option('display.max_columns', None)

prophet = StockProphet(code='035420', lag_shift=5, sequence_length=10)
for trainstartdate in pd.date_range(start='2015-01-01', periods=5 , freq='B'):
    prophet.evaluate(model_name='GradientBoostingClassifier', trainstartdate=trainstartdate, teststartdate='2019-10-01', code=None, lag_shift=None, comment=None, visual_on=False)
prophet.evaluation
```

### simulate
```python
import pandas as pd
from ailever.forecast import StockProphet
pd.set_option('display.max_columns', None)

prophet = StockProphet(code='035420', lag_shift=5, sequence_length=10)
prophet.simulate(model_name='GradientBoostingClassifier', code='035420', min_lag=5, max_lag=10, sequence_length=10, trainstartdate='2015-03-01', invest_begin='2021-10-01')
prophet.evaluation
prophet.accounts
prophet.report
```

### analyze
```python
```

### forecast
```python
import pandas as pd
from ailever.forecast import StockProphet
pd.set_option('display.max_columns', None)

prophet = StockProphet(code='035420', lag_shift=5, sequence_length=10)
prophet.evaluate(model_name='GradientBoostingClassifier', trainstartdate='2015-03-01', teststartdate='2019-10-01', code=None, lag_shift=5, comment=None, visual_on=False)
prophet.forecast(model_name='GradientBoostingClassifier', comment=None, visual_on=True)
```


### observe
```python
```
