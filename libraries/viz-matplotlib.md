## [Visualization] | [matplotlib](https://matplotlib.org/) | [github](https://github.com/matplotlib/matplotlib)

## Matplotlib-Basic
```python
import matplotlib as mpl

mpl.rcParams.keys()
```

### Korean font
```python
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumBarunGothic'
```

### Grid
```python
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(25,7)); layout=(4,2)
plt.subplot2grid(layout, (0,0), fig=fig).plot(np.random.normal(size=(100,)))
plt.subplot2grid(layout, (0,1), fig=fig).plot(np.random.normal(size=(100,)))
plt.subplot2grid(layout, (1,0), fig=fig, colspan=2).plot(np.random.normal(size=(100,)))
plt.subplot2grid(layout, (2,0), fig=fig, colspan=2, rowspan=2).plot(np.random.normal(size=(100,)))
```


## Matplotlib-Application
### Direction Fields
```python
#%%
import numpy as np
import matplotlib.pyplot as plt
import sympy

#%%
x = sympy.symbols("x")
y = sympy.Function("y")
f = y(x)**2 + x
f = sympy.lambdify((x, y(x)), f)

grid_x = np.linspace(-5, 5, 20)
grid_y = np.linspace(-5, 5, 20) 

dx = grid_x[1] - grid_x[0]
dy = grid_y[1] - grid_y[0]


for x in grid_x:
    for y in grid_y:
        # df/dx = f(x,y)
        # vector field : x*[x_unit_vector] + f(x,y)*[y_unit_vector]
        Dy = f(x, y) * dx
        cos_t = dx / (np.sqrt(dx**2 + Dy**2))
        sin_t = Dy / (np.sqrt(dx**2 + Dy**2))
        
        Dx = dx*cos_t
        Dy = dy*sin_t
        plt.plot([x-Dx/2, x+Dx/2], [y-Dy/2, y+Dy/2], 'b', lw=0.5)
```
![image](https://user-images.githubusercontent.com/52376448/99900409-acef3e00-2cf2-11eb-81e6-1abc242bf9e7.png)


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
`trajectory-2d`
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
`trajectory-3d`
```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y = np.mgrid[-3:3:30j, -5:5:30j]
F = lambda x,y : np.exp(-x**2-y**2)

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')

for i in x:
    axes.clear()
    axes.plot_surface(x,y, F(x,y), alpha=0.87)
    axes.plot(i, i, F(i,i)+0.01, marker='*', c='r')
    axes.set_xlim(-3,3)
    axes.set_ylim(-5,5)
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
