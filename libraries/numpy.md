## [Numerical Analysis] | [numpy](https://numpy.org/doc/stable/contents.html) | [github](https://github.com/numpy/numpy)

### correlation
```python
import numpy as np

x = np.linspace(0,10, 100)
f = lambda x : 2*x + np.random.normal(0, 10, size=(100))
np.corrcoef(np.c_[x, f(x)].T)
```
