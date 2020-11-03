## [Time Series] | [statsmodels](https://www.statsmodels.org/stable/api.html) | [github](https://github.com/statsmodels/statsmodels)


## Regression
### univariate

```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

x = np.linspace(-5, 5, 1000)
f = lambda x : 3*x + 5 + np.random.normal(0, 1, size=1000)
y_target = f(x)
x_input = np.stack([np.ones(1000), x], axis=1)

model = sm.OLS(y_target, x_input)
fitted_model = model.fit()

w0, w1 = fitted_model.params

"""
prstd, iv_l, iv_u = wls_prediction_std(fitted_model)
plt.plot(x, iv_u, 'r--')
plt.plot(x, iv_l, 'r--')
"""
plt.plot(x, y_target, lw=0, marker='x')
plt.plot(x, w0 + w1*x)                          # plt.plot(x, fitted_model.fittedvalues)
plt.grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97973446-6ac89000-1e09-11eb-8479-29fd28fe0160.png)

### multivariate
```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std

size = 50
weight = [10, 5, -0.02, -10.]
x = np.linspace(0, 20, size)
x_input = np.stack([x, (x-5)**2, np.sin(x), np.ones(size)], axis=1)
y_target = np.dot(x_input, weight) + np.random.normal(0, 100, size=size)

model = sm.OLS(y_target, x_input)
fitted_model = model.fit()
prstd, iv_l, iv_u = wls_prediction_std(fitted_model)

w0, w1, w2, w3 = fitted_model.params

plt.plot(x, y_target, lw=0, marker='x')
plt.plot(x, w0*x_input[:,0] + w1*x_input[:,1] + w2*x_input[:,2] + w3*x_input[:,3])      #plt.plot(x, fitted_model.fittedvalues)
plt.plot(x, iv_u, 'r--')
plt.plot(x, iv_l, 'r--')

plt.grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97973401-571d2980-1e09-11eb-91fb-f2dbe3949305.png)

<br><br><br>
## Time Series
### Stationarity
```python
import numpy as np
import statsmodels.tsa.api as smt

def stationary(series):
    """
    Augmented Dickey-Fuller test

    Null Hypothesis (H0): [if p-value > 0.5, non-stationary]
    >   Fail to reject, it suggests the time series has a unit root, meaning it is non-stationary.
    >   It has some time dependent structure.
    Alternate Hypothesis (H1): [if p-value =< 0.5, stationary]
    >   The null hypothesis is rejected; it suggests the time series does not have a unit root, meaning it is stationary.
    >   It does not have time-dependent structure.
    """
    result = smt.adfuller(series)

    print(f'[ADF Statistic] : {result[0]}')
    print(f'[p-value] : {result[1]}')
    if result[1] <= 0.05:
        print('  >>> The time series is stationary!')
    else:
        print('  >>> The time series is not stationary!')

    for key, value in result[4].items():
        print(f'[Critical Values {key} ] : {value}')

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)

a = 0.1
for t, noise in enumerate(white_noise):
    time_series[t] = a*time_series[t-1] + noise

stationary(time_series)
```
<br><br><br>

### Stationary process(1)
```python
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt

arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])

ar = np.r_[1, -arparams] # add zero-lag and negate
ma = np.r_[1, maparams] # add zero-lag

# generate samples
num_samples = 250
time_series = smt.arma_generate_sample(ar, ma, nsample=num_samples)

_, axes = plt.subplots(3,1, figsize=(10,8))
axes[0].plot(time_series)
smt.graphics.plot_acf(time_series, ax=axes[1])
smt.graphics.plot_pacf(time_series, ax=axes[2])
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97970334-ba588d00-1e04-11eb-9915-c7182e775cb9.png)

<br><br><br>
### Stationary Process(2) with numpy
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt

white_noise = np.random.normal(size=1000)
time_series1 = np.empty_like(white_noise)
time_series2 = np.empty_like(white_noise)

# AR
a = 0.9
for t, noise in enumerate(white_noise):
    time_series1[t] = a*time_series1[t-1] + noise

# MA
b = 0.9
for t, noise in enumerate(white_noise):
    time_series2[t] = noise + b*white_noise[t-1]

time_series1 = pd.Series(time_series1)
time_series2 = pd.Series(time_series2)

_, axes = plt.subplots(4,1, figsize=(10,8))
smt.graphics.plot_acf(time_series1, ax=axes[0])
smt.graphics.plot_pacf(time_series1, ax=axes[1])
smt.graphics.plot_acf(time_series2, ax=axes[2])
smt.graphics.plot_pacf(time_series2, ax=axes[3])
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97954089-e31d5a00-1de5-11eb-90c9-8ae4c7da0cda.png)
<br><br><br>


<br><br><br>
### Decompose
```python
import numpy as np
import statsmodels.tsa.api as smt

t = np.linspace(0,10,100)
f = lambda x : 2*x + np.random.normal(size=100)
series = f(t)

result = smt.seasonal_decompose(series, model='additive', freq=10)
result.plot()

observed = result.observed
trend = result.trend
seasonal = result.seasonal
resid = result.resid
```
![image](https://user-images.githubusercontent.com/52376448/97969742-d3ad0980-1e03-11eb-9049-238eeaa7c3dc.png)

<br><br><br>
### Evaluation
`ARMA`
```python
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.api as smt
import warnings
warnings.filterwarnings("ignore")

rv = np.random.RandomState(np.random.randint(0, 2020))
white_noise = rv.normal(0, 1, size=1000)
random_walk = np.cumsum(white_noise)

# evaluate an ARMA model for a given order (p,d,q)
def evaluate_arma_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = smt.ARMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out-of-sample error
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return rmse

# evaluate combinations of p and q values for an ARMA model
def evaluate_models(dataset, p_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for q in q_values:
            order = (p,q)
            try:
                rmse = evaluate_arma_model(dataset, order)
                if rmse < best_score:
                    best_score, best_cfg = rmse, order
                print('ARMA%s RMSE=%.3f' % (order,rmse))
            except:
                continue
    print('Best ARMA%s RMSE=%.3f' % (best_cfg, best_score))

# Make a prediction give regression coefficients and lag obs
def predict(coef, history):
    yhat = coef[0]
    for i in range(1, len(coef)):
        yhat += coef[i] * history[-i]
    return yhat


#%% evaluation : order (p,d,q)
white_noise = np.random.normal(size=100)
time_series = np.empty_like(white_noise)
for t, noise in enumerate(white_noise):
    time_series[t] = time_series[t-1] + noise
time_series = np.array(time_series)

# evaluate parameters
p_values = range(0, 3)
q_values = range(0, 3)
evaluate_models(time_series, p_values, q_values)
```
![image](https://user-images.githubusercontent.com/52376448/97973061-d65e2d80-1e08-11eb-87ef-8e6926cbf0bc.png)

`ARIMA`
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.api as smt
import warnings
warnings.filterwarnings("ignore")

rv = np.random.RandomState(np.random.randint(0, 2020))
white_noise = rv.normal(0, 1, size=1000)
random_walk = np.cumsum(white_noise)

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = smt.ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
    # calculate out-of-sample error
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# Make a prediction give regression coefficients and lag obs
def predict(coef, history):
    yhat = coef[0]
    for i in range(1, len(coef)):
        yhat += coef[i] * history[-i]
    return yhat


#%% evaluation : order (p,d,q)
time_series = [i + rv.normal(0,10) for i in range(1,500)]
time_series = np.array(time_series)

# evaluate parameters
p_values = range(0, 3)
d_values = range(0, 3)
q_values = range(0, 3)
evaluate_models(time_series, p_values, d_values, q_values)
```
![image](https://user-images.githubusercontent.com/52376448/97972038-48ce0e00-1e07-11eb-91e0-8f229e2cb7f7.png)

<br><br><br>

## Sampling with numpy
### White Noise and Random Walks
`white noise`
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = pd.Series(white_noise)

_, axes = plt.subplots(3,1, figsize=(12,10))
axes[0].plot(time_series)
pd.plotting.lag_plot(time_series, lag=3, ax=axes[1])
pd.plotting.autocorrelation_plot(time_series, ax=axes[2])

axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96566111-2369dc00-1300-11eb-8d66-d89c4df8617a.png)

`random walks`
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)

for t, noise in enumerate(white_noise):
    time_series[t] = time_series[t-1] + noise

time_series = pd.Series(time_series)

_, axes = plt.subplots(3,1, figsize=(12,10))
axes[0].plot(time_series)
pd.plotting.lag_plot(time_series, lag=3, ax=axes[1])
pd.plotting.autocorrelation_plot(time_series, ax=axes[2])

axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96565857-ddad1380-12ff-11eb-8d30-02392443cc08.png)

<br><br><br>
### Linear Models
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)  # linear

b0 = -.1
b1 = .002
for t, noise in enumerate(white_noise):
    time_series[t] = b0 + b1*t + noise

time_series = pd.Series(time_series)

_, axes = plt.subplots(3,1, figsize=(12,10))
axes[0].plot(time_series)
pd.plotting.lag_plot(time_series, lag=3, ax=axes[1])
pd.plotting.autocorrelation_plot(time_series, ax=axes[2])

axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96568173-9a07d900-1302-11eb-94d9-dfb390239110.png)

<br><br><br>
### Log Models
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)  # linear

b0 = -.1
b1 = 1
b2 = 1
for t, noise in enumerate(white_noise):
    time_series[t] = b0 + b1*np.log(b2*t) + noise

time_series = pd.Series(time_series[1:])

_, axes = plt.subplots(3,1, figsize=(12,10))
axes[0].plot(time_series)
pd.plotting.lag_plot(time_series, lag=3, ax=axes[1])
pd.plotting.autocorrelation_plot(time_series, ax=axes[2])

axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96567007-3b8e2b00-1301-11eb-925a-143bd868c181.png)

<br><br><br>

### Exponential Models
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

white_noise = np.random.normal(0, 5, size=1000)
time_series = np.empty_like(white_noise)  # linear

b0 = -.1
b1 = 0.001
b2 = 0.01
for t, noise in enumerate(white_noise):
    time_series[t] = b0 + b1*np.exp(b2*t) + noise

time_series = pd.Series(time_series)

_, axes = plt.subplots(3,1, figsize=(12,10))
axes[0].plot(time_series)
pd.plotting.lag_plot(time_series, lag=3, ax=axes[1])
pd.plotting.autocorrelation_plot(time_series, ax=axes[2])

axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96569041-99237700-1303-11eb-85f3-4d7e36c5efac.png)

<br><br><br>

### Cos Models
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)  # linear

b0 = -.1
b1 = 1
b2 = 1
for t, noise in enumerate(white_noise):
    time_series[t] = b0 + b1*np.cos(b2*t) + noise

time_series = pd.Series(time_series)

_, axes = plt.subplots(3,1, figsize=(12,10))
axes[0].plot(time_series)
pd.plotting.lag_plot(time_series, lag=3, ax=axes[1])
pd.plotting.autocorrelation_plot(time_series, ax=axes[2])

axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96570083-e0f6ce00-1304-11eb-9c3d-1204ca6cb362.png)
<br><br><br>

### Autoregressive Models - AR(p)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)  # linear

a = 0.5
for t, noise in enumerate(white_noise):
    time_series[t] = a*time_series[t-1] + noise

time_series = pd.Series(time_series)

_, axes = plt.subplots(3,1, figsize=(12,10))
axes[0].plot(time_series)
pd.plotting.lag_plot(time_series, lag=3, ax=axes[1])
pd.plotting.autocorrelation_plot(time_series, ax=axes[2])

axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96568459-efdc8100-1302-11eb-9c7e-9f9270e762b4.png)


<br><br><br>
### Moving Average Models - MA(q)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

white_noise = np.random.normal(0, 5, size=1000)
time_series = np.empty_like(white_noise)  # linear

b = 0.5
for t, noise in enumerate(white_noise):
    time_series[t] = noise + b*white_noise[t-1]

time_series = pd.Series(time_series)

_, axes = plt.subplots(3,1, figsize=(12,10))
axes[0].plot(time_series)
pd.plotting.lag_plot(time_series, lag=3, ax=axes[1])
pd.plotting.autocorrelation_plot(time_series, ax=axes[2])

axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96569452-1ea72700-1304-11eb-82fd-116cf33471ee.png)

<br><br><br>
### Autoregressive Moving Average Models - ARMA(p, q)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)  # linear

a = 0.1
b = 0.09
for t, noise in enumerate(white_noise):
    time_series[t] = a*time_series[t-1] + noise + b*white_noise[t-1]

time_series = pd.Series(time_series)

_, axes = plt.subplots(3,1, figsize=(12,10))
axes[0].plot(time_series)
pd.plotting.lag_plot(time_series, lag=3, ax=axes[1])
pd.plotting.autocorrelation_plot(time_series, ax=axes[2])

axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96570810-c40eca80-1305-11eb-8581-3d9289f41aae.png)

<br><br><br>

### Autoregressive Conditionally Heterskedastic Models - ARCH(p)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)  # linear

b0 = 0.01
b1 = 0.3
for t, noise in enumerate(white_noise):
    time_series[t] = noise * np.sqrt((b0 + b1*white_noise[t-1]**2))

time_series = pd.Series(time_series)

_, axes = plt.subplots(3,1, figsize=(12,10))
axes[0].plot(time_series)
pd.plotting.lag_plot(time_series, lag=3, ax=axes[1])
pd.plotting.autocorrelation_plot(time_series, ax=axes[2])

axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96571281-4eefc500-1306-11eb-951b-fb24063ebe07.png)

<br><br><br>
### Generalized Autoregressive Conditionally Heterskedastic Models - GARCH(p, q)
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
sigma2 = np.empty_like(white_noise)
time_series = np.empty_like(white_noise)

b0 = 0.01
b1 = 0.9
b2 = 0.01
for t, noise in enumerate(white_noise):
    sigma2[t] = b0 + b1*white_noise[t-1]**2 + b2*sigma2[t-1]
    time_series[t] = noise * np.sqrt(sigma2[t])

time_series = pd.Series(time_series)

_, axes = plt.subplots(3,1, figsize=(12,10))
axes[0].plot(time_series)
pd.plotting.lag_plot(time_series, lag=3, ax=axes[1])
pd.plotting.autocorrelation_plot(time_series, ax=axes[2])

axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.tight_layout()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96571873-20beb500-1307-11eb-8395-2fdccf3106ad.png)

