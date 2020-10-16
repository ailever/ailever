## [Time Series] | [statsmodels](https://www.statsmodels.org/stable/api.html) | [github](https://github.com/statsmodels/statsmodels)

### Stationarity
```python

```

<br><br><br>
## Sampling
### White Noise and Random Walks
`white noise`
```python
import numpy as np
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
plt.plot(white_noise)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96222992-2d21d580-0fc8-11eb-840a-8c422fbe650a.png)

`random walks`
```python
import numpy as np
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)

for t, noise in enumerate(white_noise):
    time_series[t] = time_series[t-1] + noise
plt.plot(time_series)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96223026-3b6ff180-0fc8-11eb-9dd6-2ce0be02ab9f.png)

<br><br><br>
### Linear Models
```python
import numpy as np
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)  # linear

b0 = -.1
b1 = .03
for t, noise in enumerate(white_noise):
    time_series[t] = b0 + b1*t + noise
plt.plot(time_series)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96223798-7888b380-0fc9-11eb-8bdb-98965c619a6f.png)

<br><br><br>
### Log Models
```python
import numpy as np
import matplotlib.pyplot as plt

white_noise = np.random.normal(size=1000)
time_series = np.empty_like(white_noise)

b0 = -.1
b1 = 3
for t, noise in enumerate(white_noise):
    time_series[t] = b0 + b1*np.log(t) + noise
plt.plot(time_series)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/96224639-e1bcf680-0fca-11eb-83bd-40f490ba8795.png)

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


```

<br><br><br>
### Generalized Autoregressive Conditionally Heterskedastic Models - GARCH(p, q)
```python
import numpy as np
import matplotlib.pyplot as plt


```

