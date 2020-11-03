## [Numerical Analysis] | [scipy](https://docs.scipy.org/doc/scipy/reference/) | [github](https://github.com/scipy/scipy)

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
```

`Repeated Measures ANOVA Test`
```python
```
