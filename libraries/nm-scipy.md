## [Numerical Analysis] | [scipy](https://docs.scipy.org/doc/scipy/reference/) | [github](https://github.com/scipy/scipy) | [handbook](https://scipy-cookbook.readthedocs.io/index.html)

## Hypothesis Test
### Normality Test
```python
import numpy as np
from scipy import stats

data = 5 * np.random.normal(size=100)

stat, p = stats.shapiro(data) # Shapiro–Wilk test
stat, p = stats.kstest(data, 'norm') # Kolmogorov–Smirnov test
stat, p = stats.normaltest(data) # D'Agostino's K-squared test
result = stats.anderson(data) # Anderson–Darling test
#result.statistic
#result.critical_values
#result.significance_level

result = stats.jarque_bera(data) # Jarque–Bera test
#result.statistic
#result.pvalue
```




## [Parametric method]

### Box-Cox Transformation 
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import boxcox, inv_boxcox

lmbda=0.1
raw_data = np.random.normal(0,1, size=100)
raw_data = np.exp(raw_data)
transformed_data = stats.boxcox(raw_data, lmbda=lmbda)

# for inverse
transformed_data_ = boxcox(raw_data, lmbda)
inv_transformed_data = inv_boxcox(transformed_data_, lmbda)

_, axes = plt.subplots(3,1)
axes[0].hist(raw_data, bins=30)
axes[1].hist(transformed_data, bins=30)
axes[2].hist(inv_transformed_data, bins=30)
axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97964912-df490200-1dfc-11eb-9a0c-14685836b9a4.png)

### Correlation
`Pearson's correlation`
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# prepare data
data1 = 5*np.random.normal(size=100) + 50
data2 = data1 + np.random.normal(0,5, size=100)

# calculate Pearson's correlation
corr, p = stats.pearsonr(data1, data2)
# display the correlation
print('Pearsons correlation: %.3f' % corr)

# interpret the significance
alpha = 0.05
if p > alpha:
    print('No correlation (fail to reject H0)')
else:
    print('Some correlation (reject H0)')

plt.scatter(data1, data2)
plt.grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97961369-85453e00-1df6-11eb-9393-c4876f0fcd7d.png)

### Significance Tests
`Student’s t-Test`
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data1 = 5 * np.random.normal(size=100) + 50
data2 = 5 * np.random.normal(size=100) + 51

stat, p = stats.ttest_ind(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')

_, axes = plt.subplots(2,1)
axes[0].hist(data1, bins=30)
axes[1].hist(data2, bins=30)
axes[0].grid(True)
axes[1].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97956598-6e99e980-1dec-11eb-90d3-a25127cdc722.png)

`Paired Student’s t-Test`
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data1 = 5 * np.random.normal(size=100) + 50
data2 = data1 + np.random.normal(1, 1, size=100)

stat, p = stats.ttest_rel(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))

alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')

_, axes = plt.subplots(2,1)
axes[0].hist(data1, bins=30)
axes[1].hist(data2, bins=30)
axes[0].grid(True)
axes[1].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97956636-86716d80-1dec-11eb-987f-5eb86bb940ef.png)


`Analysis of Variance Test`
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# generate three independent samples
data1 = 5 * np.random.normal(size=100) + 50
data2 = 5 * np.random.normal(size=100) + 50
data3 = 5 * np.random.normal(size=100) + 52

# compare samples
stat, p = stats.f_oneway(data1, data2, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Same distributions (fail to reject H0)')
else:
    print('Different distributions (reject H0)')

_, axes = plt.subplots(3,1)
axes[0].hist(data1, bins=30)
axes[1].hist(data2, bins=30)
axes[2].hist(data3, bins=30)
axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97956696-af91fe00-1dec-11eb-8bb4-6db2a4c553b8.png)

### Normality Tests
`Shapiro-Wilk Test`
> when p < alpha, reject : not look Gaussian!

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

data = 5 * np.random.normal(size=100) + 50

# normality test
stat, p = stats.shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

_, axes = plt.subplots(2,1)
axes[0].hist(data, bins=30)
sm.graphics.qqplot(data, ax=axes[1])
axes[0].grid(True)
axes[1].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97960093-136bf500-1df4-11eb-993c-1e31cca23b7c.png)

`D’Agostino’s K2 Test`
> when p < alpha, reject : not look Gaussian!

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

data = 5 * np.random.normal(size=100) + 50

# normality test
stat, p = stats.normaltest(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Sample looks Gaussian (fail to reject H0)')
else:
    print('Sample does not look Gaussian (reject H0)')

_, axes = plt.subplots(2,1)
axes[0].hist(data, bins=30)
sm.graphics.qqplot(data, ax=axes[1])
axes[0].grid(True)
axes[1].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97960009-e7e90a80-1df3-11eb-89d9-d986346a9c7e.png)

`Kolmogorov–Smirnov test`
> when p < alpha, reject : not look Gaussian!

```python
```

`Lilliefors test`
> when p < alpha, reject : not look Gaussian!

```python
```


`Anderson–Darling test`
> when p < alpha, reject : not look Gaussian!

```python
```

`Jarque–Bera test`
> when p < alpha, reject : not look Gaussian!

```python
```

`Pearson's chi-squared test`
> when p < alpha, reject : not look Gaussian!

```python
```


### Independence Test
`Chi-Squared Test`
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# contingency table
table = np.array([[10,20,30],
                  [20,40,60]])
stat, p, dof, expected = stats.chi2_contingency(table)
print('degree of freedom = %d' % dof)
print(expected, '\n')

# interpret test-statistic
prob = 0.95
critical = stats.chi2.ppf(prob, dof)
print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, stat))
if abs(stat) >= critical:
    print('Dependent (reject H0)\n')
else:
    print('Independent (fail to reject H0)\n')

# interpret p-value
alpha = 1.0 - prob
print('significance=%.3f, p=%.3f' % (alpha, p))
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')

plt.pcolor(table)
plt.colorbar()
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97959449-bfacdc00-1df2-11eb-89f6-50c5665a8593.png)

## [Nonparametric method]
### Rank correlation
`Spearman’s Rank Correlation`
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# prepare data
x = np.linspace(0,10,100)
f = lambda x : 2*x + np.random.normal(0,5, size=100)

# calculate kendalltau's correlation
coef, p = stats.spearmanr(x, f(x))
print('Spearmans correlation coefficient: %.3f' % coef)

# interpret the significance
alpha = 0.05
if p > alpha:
    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
    print('Samples are correlated (reject H0) p=%.3f' % p)

plt.scatter(x,f(x))
plt.grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97960835-7f9b2880-1df5-11eb-9937-fc6275b1daad.png)

`Kendall’s Rank Correlation`
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# prepare data
x = np.linspace(0,10,100)
f = lambda x : 2*x + np.random.normal(0,5, size=100)

# calculate spearman's correlation
coef, p = stats.kendalltau(x, f(x))
print('Kendall’s correlation coefficient: %.3f' % coef)

# interpret the significance
alpha = 0.05
if p > alpha:
    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
else:
    print('Samples are correlated (reject H0) p=%.3f' % p)

plt.scatter(x,f(x))
plt.grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97960921-a8bbb900-1df5-11eb-88f7-fc108ad3586c.png)

### Rank Significance Tests
`Mann-Whitney U Test`
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data1 = 50 + (np.random.rand(100) * 10)
data2 = 51 + (np.random.rand(100) * 10)

# compare samples
stat, p = stats.mannwhitneyu(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')

_, axes = plt.subplots(2,1)
axes[0].hist(data1, bins=30)
axes[1].hist(data2, bins=30)
axes[0].grid(True)
axes[1].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97962064-c1c56980-1df7-11eb-8839-cec4009d728a.png)

`Wilcoxon Signed-Rank Test`
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data1 = 50 + (np.random.rand(100) * 10)
data2 = 51 + (np.random.rand(100) * 10)

# compare samples
stat, p = stats.wilcoxon(data1, data2)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')

_, axes = plt.subplots(2,1)
axes[0].hist(data1, bins=30)
axes[1].hist(data2, bins=30)
axes[0].grid(True)
axes[1].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97962107-d9045700-1df7-11eb-820f-66262074c379.png)

`Kruskal-Wallis H Test`
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data1 = 50 + (np.random.rand(100) * 10)
data2 = 51 + (np.random.rand(100) * 10)
data3 = 52 + (np.random.rand(100) * 10)

# compare samples
stat, p = stats.kruskal(data1, data2, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')

_, axes = plt.subplots(3,1)
axes[0].hist(data1, bins=30)
axes[1].hist(data2, bins=30)
axes[2].hist(data3, bins=30)
axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97962244-0fda6d00-1df8-11eb-8704-0cd71dfdfa6f.png)

`Friedman Test`
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

data1 = 50 + (np.random.rand(100) * 10)
data2 = data1 + np.random.normal(size=100)
data3 = data2 + np.random.normal(size=100)

# compare samples
stat, p = stats.friedmanchisquare(data1, data2, data3)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
alpha = 0.05
if p > alpha:
    print('Same distribution (fail to reject H0)')
else:
    print('Different distribution (reject H0)')

_, axes = plt.subplots(3,1)
axes[0].hist(data1, bins=30)
axes[1].hist(data2, bins=30)
axes[2].hist(data3, bins=30)
axes[0].grid(True)
axes[1].grid(True)
axes[2].grid(True)
plt.show()
```
![image](https://user-images.githubusercontent.com/52376448/97962758-2af9ac80-1df9-11eb-8152-8a3c2e54fd4d.png)




## Numerical Method
### Curve Fit
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
rng = np.random.default_rng()
y_noise = 0.2 * rng.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')

popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
plt.plot(xdata, func(xdata, *popt), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
```

### Distribution Fit
```python
from scipy.stats import norm
import matplotlib.pyplot as plt

param1 = 1.
param2 = 2.
x = norm.rvs(param1, param2, size=1000, random_state=123)
param = norm.fit(x)
plt.plot(norm.pdf(np.linspace(-1, 3, 1000), *param))
```
