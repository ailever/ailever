## [Visualization] | [matplotlib](https://matplotlib.org/) | [github](https://github.com/matplotlib/matplotlib)

### Animate with IPython.display
`one figure`
```python
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

for _ in range(10):
    plt.clf()
    plt.plot(np.random.normal(size=1000))
    plt.ylim(-3,3)
    plt.grid()
    display.display(plt.gcf())
    display.clear_output(wait=True)
```
`several figure`
```python
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

_, axes = plt.subplots(2,1)
for _ in range(10):
    axes[0].clear()
    axes[1].clear()
    axes[0].plot(np.random.normal(0,1, size=100))
    axes[1].plot(np.random.normal(0,1, size=100))
    display.display(plt.gcf())
    display.clear_output(wait=True)
```

### UI Interface with ipywidgets
`one figure`
```python
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

@interact(x=(-10,10,1))
def f(x):
    plt.plot(np.arange(0,10), x*np.arange(0,10))
```
`several figure`
```python
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

@interact(x=(-10,10,1))
def f(x):
    _, axes = plt.subplots(2,1, figsize=(10,8))
    axes[0].plot(np.arange(0,10), x*np.arange(0,10))
    axes[1].plot(np.arange(0,10), np.arange(0,10)**abs(x))
    axes[0].set_ylim([-20,20])
    axes[1].set_ylim([-20,20])
```
