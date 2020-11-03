## [Numerical Analysis] | [scipy](https://docs.scipy.org/doc/scipy/reference/) | [github](https://github.com/scipy/scipy)
## [Parametric method]
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

# calculate spearman's correlation
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
![image](https://user-images.githubusercontent.com/52376448/97960921-a8bbb900-1df5-11eb-88f7-fc108ad3586c.png)

