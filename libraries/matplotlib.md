## [Visualization] | [matplotlib](https://matplotlib.org/) | [github](https://github.com/matplotlib/matplotlib)
### 3D plot
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y = np.mgrid[-3:3:100j, -5:5:100j]
F = lambda x,y : np.exp(-x**2-y**2)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
axes.plot_surface(x,y, F(x,y))
plt.show()
```

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
`several figures`
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
`trajectory`
```python
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

x = np.arange(-10,11)
y = lambda x : x**2

for i in x:
    plt.clf()
    plt.plot(x, y(x))
    plt.plot(i, y(i), marker='o')
    plt.xlim(-12,12)
    plt.ylim(-1,120)
    plt.grid(True)
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
`several figures`
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
`trajectory`
```python
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

x = np.arange(-10,11)
y = lambda x : x**2

@interact(i=(-10,10,1))
def f(i):
    plt.plot(x, y(x))
    plt.plot(i, y(i), marker='o')
```
