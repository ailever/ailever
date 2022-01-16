## [Visualization] | [matplotlib](https://matplotlib.org/) | [github](https://github.com/matplotlib/matplotlib)

## Matplotlib-Basic
- https://matplotlib.org/2.0.2/examples/color/named_colors.html
- https://matplotlib.org/2.0.2/examples/lines_bars_and_markers/marker_reference.html
- https://matplotlib.org/2.0.2/examples/lines_bars_and_markers/line_styles_reference.html
- https://codetorial.net/matplotlib/index.html  
- https://userdyk-github.github.io/pl03-topic02/PL03-Topic02-Matplotlib.html
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

### FillBetween
```python
import matplotlib.pyplot as plt

plt.plot([1,2,3])
ylim = plt.ylim()
plt.fill_between(facecolor='k', alpha=0.1,
    x = [0, 1, 2, 3, 4, 5], 
    y1 = ylim[0], 
    y2 = ylim[1], 
    where = [0,1,0,1,1,1])
```
![image](https://user-images.githubusercontent.com/56889151/149651189-69471a55-fb66-46dc-9737-1e214590648b.png)

### Double Axis
```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax0 = plt.subplots()
ax1 = ax0.twinx()
lns0 = ax0.plot([10, 5, 2, 9, 7], 'r-', label="y0")
lns1 = ax1.plot([100, 200, 220, 180, 120], 'g:', label="y1")

lns = lns0+lns1
labs = [l.get_label() for l in lns]
ax0.legend(lns, labs, loc=0)

ax0.set_title("Plot")
ax0.set_xlabel("sharing x-axis")
ax0.set_ylabel("y0")
ax0.grid(False)
ax1.set_ylabel("y1")
ax1.grid(False)
plt.show()
```
![image](https://user-images.githubusercontent.com/56889151/149651636-4e18d8e1-e90d-434b-b3a4-23f1bc7be647.png)

### Set Ticks
`Figure`
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 50)
y = np.cos(x)

plt.plot(x, y)
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
           [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
plt.yticks([-1, 0, 1], ["Low", "Zero", "High"])
plt.show()
```
`Axes`
```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 50)
y = np.cos(x)

_, axes = plt.subplots(1,1)
axes.plot(x, y)
axes.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
axes.set_yticks([-1, 0, 1])
xticks = axes.get_xticks().tolist()
yticks = axes.get_yticks().tolist()

axes.xaxis.set_major_locator(mpl.ticker.FixedLocator(xticks))
axes.yaxis.set_major_locator(mpl.ticker.FixedLocator(yticks))
axes.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])
axes.set_yticklabels(["Low", "Zero", "High"])
plt.show()
```
![image](https://user-images.githubusercontent.com/56889151/149652014-837f56fb-e52e-42a1-a721-c3bd97590a48.png)

### Text & Annotation
```python
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-4, 4, 1024)
Y = .25 * (X + 4.) * (X + 1.) * (X - 2.)

plt.plot(X, Y, c = 'k')
plt.text(-0.5, -0.25, 'Brackmard minimum')
plt.annotate("Annotation", xy=(-4, 0), xytext=(+20, +50), fontsize=14, family="serif", xycoords="data", textcoords="offset points", arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.5"))

plt.show()
```
![image](https://user-images.githubusercontent.com/56889151/149652332-5191d71e-bc49-40ae-a345-f56b01369ace.png)


### Insets
```python
import numpy as np
import matplotlib.pyplot as plt

# main graph
X = np.linspace(-6, 6, 1024)
Y = np.sinc(X)
plt.plot(X, Y, c = 'k')
plt.axvline(0, ymax=0.3, color="grey", linestyle=":")
plt.axvline(2, ymax=0.3, color="grey", linestyle=":")

# inset
x = np.linspace(0, 2, 1024)
y = np.sinc(x)

left_bottom_point_xposition_ratio = 0.6
left_bottom_point_yposition_ratio = 0.6
width_ratio = .25
height_ratio = .25
sub_axes = plt.axes([left_bottom_point_xposition_ratio, left_bottom_point_yposition_ratio, width_ratio, height_ratio])
sub_axes.plot(x, y, c = 'k')
#plt.setp(sub_axes)
plt.show()
```
![image](https://user-images.githubusercontent.com/56889151/149652825-f71c38a2-7b46-4e3e-b154-d46b431f3927.png)

### Summary
`Figure`
```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 50)
y = np.cos(x)

plt.plot(x, y, label='Line')
plt.xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
plt.yticks([-1, 0, 1])
plt.xlabel('X-label')
plt.ylabel('Y-label')
plt.title('TITLE')
plt.legend()
plt.show()
```
`Axes`
```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 50)
y = np.cos(x)

_, axes = plt.subplots(1,1)
axes.plot(x, y, label='Line')
axes.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
axes.set_yticks([-1, 0, 1])
axes.set_xlabel('X-label')
axes.set_ylabel('Y-label')
axes.set_title('TITLE')
axes.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/56889151/149652150-c7285d09-7e4e-4fab-a488-4d09bed29cff.png)



---

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
