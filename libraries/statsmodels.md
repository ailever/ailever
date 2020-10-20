## [Time Series] | [statsmodels](https://www.statsmodels.org/stable/api.html) | [github](https://github.com/statsmodels/statsmodels)

### Stationarity
```python

```

<br><br><br>
## Sampling with numpy
### White Noise and Random Walks
`white noise`
```python
import numpy as np
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
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)  # linear

b0 = -.1
b1 = .0007
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
![image](https://user-images.githubusercontent.com/52376448/96566416-80fe2880-1300-11eb-9d46-a458ec08eb3d.png)

<br><br><br>
### Log Models
```python
import numpy as np
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)  # linear

b0 = -.1
b1 = 1
for t, noise in enumerate(white_noise):
    time_series[t] = b0 + b1*np.log(t) + noise

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
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
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
![image](https://user-images.githubusercontent.com/52376448/96567541-dd157c80-1301-11eb-8ba4-ca734463697a.png)

<br><br><br>

### Autoregressive Models - AR(p)
```python
import numpy as np
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)

a = 0.9
for t, noise in enumerate(white_noise):
    time_series[t] = a*time_series[t-1] + noise
plt.plot(time_series)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96225048-71fb3b80-0fcb-11eb-949e-dad8cda32b61.png)


<br><br><br>
### Moving Average Models - MA(q)
```python
import numpy as np
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)

b = 0.9
for t, noise in enumerate(white_noise):
    time_series[t] = noise + b*white_noise[t-1]
plt.plot(time_series)
plt.show()
```

<br><br><br>
### Autoregressive Moving Average Models - ARMA(p, q)
```python
import numpy as np
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)

a = 0.01
b = 0.09
for t, noise in enumerate(white_noise):
    time_series[t] = a*time_series[t-1] + noise + b*white_noise[t-1]
plt.plot(time_series)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96225716-74aa6080-0fcc-11eb-85db-c4a36acbf8a4.png)

<br><br><br>

### Autoregressive Conditionally Heterskedastic Models - ARCH(p)
```python
import numpy as np
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)

b0 = 0.01
b1 = 0.09
for t, noise in enumerate(white_noise):
    time_series[t] = noise * np.sqrt((b0 + b1*white_noise[t-1]**2))
plt.plot(time_series)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96226263-5e50d480-0fcd-11eb-8594-ec47b91fee97.png)

<br><br><br>
### Generalized Autoregressive Conditionally Heterskedastic Models - GARCH(p, q)
```python
import numpy as np
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
sigma2 = np.empty_like(white_noise)
time_series = np.empty_like(white_noise)

b0 = 0.01
b1 = 0.09
b2 = 0.01
for t, noise in enumerate(white_noise):
    sigma2[t] = b0 + b1*white_noise[t-1]**2 + b2*sigma2[t-1]
    time_series[t] = noise * np.sqrt(sigma2[t])
plt.plot(time_series)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96227538-37939d80-0fcf-11eb-8ea2-4517ec42490a.png)

