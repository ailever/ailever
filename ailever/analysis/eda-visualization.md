## Matplotlib
`subplot2grid`
```python
from matplotlib import pyplot as plt
plt.style.use('seaborn-notebook') # plt.style.available

gridcols = 4
table_columns = 10
quotient = table_columns//gridcols
reminder = table_columns%gridcols
layout = (quotient, gridcols) if reminder==0 else (quotient+1, gridcols)
fig = plt.figure(figsize=(25, layout[0]*5))
axes = dict()
for i in range(0, layout[0]):
    for j in range(0, layout[1]):
        idx = i*layout[1] + j
        axes[idx]= plt.subplot2grid(layout, (i, j))
```

### Installation
### Numerical Variables
### Categorical Variables

<br><br><br>

---

## Seaborn
### Installation
### Numerical Variables
### Categorical Variables

<br><br><br>

---


## Plotly
### Installation
### Numerical Variables
### Categorical Variables

<br><br><br>

---


## Dash
### Installation
### Numerical Variables
### Categorical Variables

<br><br><br>

---

