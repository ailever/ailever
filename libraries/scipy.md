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
![image](https://user-images.githubusercontent.com/52376448/97955822-7a84ac00-1dea-11eb-9fa4-0af02edd87be.png)

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
![image](https://user-images.githubusercontent.com/52376448/97956394-e4ea1c00-1deb-11eb-806c-e7d4cb5429fb.png)

`Analysis of Variance Test`
```python
```

`Repeated Measures ANOVA Test`
```python
```
