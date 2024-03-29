## Sample Analysis
- https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/

### Binomial Test
```python
from scipy import stats
import numpy as np

N, mu_0 = 10, 0.5
np.random.seed(0)
x = stats.bernoulli(mu_0).rvs(N)
n = np.count_nonzero(x)
stats.binom_test(n, N)
```
```python
from scipy import stats
import numpy as np

N, mu_0 = 100, 0.5
np.random.seed(0)
x = stats.bernoulli(mu_0).rvs(N)
n = np.count_nonzero(x)
stats.binom_test(n, N)
```

### Chi-squared Test
`Goodness of fit Testing`
```python
from scipy import stats
import numpy as np

N, K = 10, 4
mu_0 = np.ones(K)/K

x = np.random.choice(K, N, p=mu_0)
n = np.bincount(x, minlength=K)
stats.chisquare(n)
```
`Independence Testing`
```python
from scipy import stats
import numpy as np

obs = np.array([[5, 15], [10, 20]])
stats.chi2_contingency(obs)
```


### Equal Mean Testing
#### Z-test
```python
from scipy import stats
import numpy as np

def ztest_1samp(x, sigma2=1, mu=0):
    z = (x.mean() - mu) / np.sqrt(sigma2/len(x))
    return z, 2 * stats.norm().sf(np.abs(z))

N, mu_0 = 10, 0

x = stats.norm(mu_0).rvs(N)
ztest_1samp(x)
```

#### T-test
```python
from scipy import stats
import numpy as np

N, mu_0 = 10, 0

x = stats.norm(mu_0).rvs(N)
stats.ttest_1samp(x, popmean=0)
```

```python
from scipy import stats

rvs1 = stats.norm.rvs(loc=5,scale=10,size=500)
rvs2 = stats.norm.rvs(loc=5,scale=10,size=500)

stats.ttest_ind(rvs1,rvs2, equal_var = True)
stats.ttest_ind(rvs1,rvs2, equal_var = False)
```

```python
from scipy import stats
import numpy as np

N_1, mu_1, sigma_1 = 50, 0, 1
N_2, mu_2, sigma_2 = 100, 0.5, 1

np.random.seed(0)
x1 = stats.norm(mu_1, sigma_1).rvs(N_1)
x2 = stats.norm(mu_2, sigma_2).rvs(N_2)
stats.ttest_ind(x1, x2, equal_var=True)
```

```python
from scipy import stats
import numpy as np

N_1, mu_1, sigma_1 = 10, 0, 1
N_2, mu_2, sigma_2 = 10, 0.5, 1

np.random.seed(0)
x1 = stats.norm(mu_1, sigma_1).rvs(N_1)
x2 = stats.norm(mu_2, sigma_2).rvs(N_2)
stats.ttest_ind(x1, x2, equal_var=False)
```

```python
from scipy import stats
import numpy as np

N = 5
mu_1, mu_2 = 0, 0.4

x1 = stats.norm(mu_1).rvs(N)
x2 = x1 + stats.norm(mu_2, 0.1).rvs(N)

stats.ttest_rel(x1, x2)
```

### Equal Variance Testing
```python
from scipy import stats
import numpy as np

N1, sigma_1 = 100, 1
N2, sigma_2 = 100, 1.2

x1 = stats.norm(0, sigma_1).rvs(N1)
x2 = stats.norm(0, sigma_2).rvs(N2)

equal_variance = dict()
equal_variance['bartlett'] = stats.bartlett(x1, x2)
equal_variance['fligner'] = stats.fligner(x1, x2)
equal_variance['levene'] = stats.levene(x1, x2)

#equal_variance['~'].statistic
#equal_variance['~'].pvalue
```


## Time Series Analysis
### Normality Testing
```python
import numpy as np
from scipy import stats

data = 5 * np.random.normal(size=100)

normality = dict()
normality['shapiro'] = stats.shapiro(data) # Shapiro–Wilk test
normality['kstest'] = stats.kstest(data, 'norm') # Kolmogorov–Smirnov test
normality['normaltest'] = stats.normaltest(data) # D'Agostino's K-squared test
normality['anderson'] = stats.anderson(data) # Anderson–Darling test
normality['jarque_bera'] = stats.jarque_bera(data) # Jarque–Bera test

#stat, p = normality['shapiro']
#stat, p = normality['kstest']
#stat, p = normality['normaltest']
#normality['anderson'].statistic
#normality['anderson'].critical_values
#normality['anderson'].significance_level
#normality['jarque_bera'].statistic
#normality['jarque_bera'].pvalue
```

### AutoCorrelation Testing
```python
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.stats import diagnostic

n_samples = 300
ar_params = np.r_[0.3, 0.1]
ma_params = np.r_[0.1, 0.1]
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]

y = smt.ArmaProcess(ar, ma).generate_sample(n_samples, burnin=50)

diagnostic.acorr_ljungbox(y, lags=range(1,10))
```

### Heteroscedasticity Testing
```python
import numpy as np
from statsmodels.stats import diagnostic

white_noise = np.random.normal(size=300)
sigma2 = np.empty_like(white_noise)
time_series = np.empty_like(white_noise)

b0 = 0.01
b1 = 0.9
b2 = 0.01
for t, noise in enumerate(white_noise):
    sigma2[t] = b0 + b1*white_noise[t-1]**2 + b2*sigma2[t-1]
    time_series[t] = noise * np.sqrt(sigma2[t])

exog = np.c_[np.ones_like(time_series), np.random.normal(0, 1, size=time_series.shape[0])]
f_stat, f_pvalue, alternative = diagnostic.het_goldfeldquandt(time_series, x=exog, alternative='two-sided')
multiplier_stat, multiplier_pvalue, f_stat, f_pvalue = diagnostic.het_white(time_series, exog=exog)
multiplier_stat, multiplier_pvalue, f_stat, f_pvalue = diagnostic.het_breuschpagan(time_series, exog_het=exog)
```

### Stationary Testing
```python
import numpy as np
import statsmodels.tsa.api as smt

n_samples = 300
ar_params = np.r_[0.3, 0.1]
ma_params = np.r_[0.1, 0.1]
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]

y = smt.ArmaProcess(ar, ma).generate_sample(n_samples, burnin=50)

stat, pvalue, lags, observation, critical_value, maximum_information_criteria = smt.stattools.adfuller(y)
stat, pvalue, lags, critical_value = smt.stattools.kpss(y)
```

### Unit Root Testing
```python
from arch.unitroot import ADF, DFGLS, PhillipsPerron, ZivotAndrews, VarianceRatio, KPSS
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as smt

n_samples = 300
ar_params = np.r_[0.3, 0.1]
ma_params = np.r_[0.1, 0.1]
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]

y = smt.ArmaProcess(ar, ma).generate_sample(n_samples, burnin=50)

unit_root_testing = dict()
unit_root_testing['adf'] = ADF(y)
unit_root_testing['dfgls'] = DFGLS(y)
unit_root_testing['pp'] = PhillipsPerron(y)
unit_root_testing['za'] = ZivotAndrews(y)
unit_root_testing['vr'] = VarianceRatio(y)
unit_root_testing['kpss'] = KPSS(y)

#unit_root_testing['~'].trend
#unit_root_testing['~'].lags
#unit_root_testing['~'].stat
#unit_root_testing['~'].pvalue
```

### Cointegration Testing
```python
from arch.unitroot import cointegration
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as smt

n_samples = 300
ar_params = np.r_[0.3, 0.1]
ma_params = np.r_[0.1, 0.1]
ar, ma = np.r_[1, -ar_params], np.r_[1, ma_params]

y1 = smt.ArmaProcess(ar, ma).generate_sample(n_samples, burnin=50)
y2 = 0.3 * y1 + np.random.normal(size=n_samples)  # schtochastic/deterministic

cointegration_testing = dict()
cointegration_testing['eg'] = cointegration.engle_granger(y1, y2)
cointegration_testing['po'] = cointegration.phillips_ouliaris(y1, y2)
cointegration_testing['coint'] = smt.coint(y1, y2) # Test for no-cointegration of a univariate equation

# statistic, pvalue, _ = cointegration_testing['coint'] 
#cointegration_testing['eg or po'].trend
#cointegration_testing['eg or po'].stat
#cointegration_testing['eg or po'].pvalue
```



