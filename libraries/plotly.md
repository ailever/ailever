## [Visualization]|[plotly](https://plotly.com/python/) | [github](https://github.com/plotly/plotly.py)
### Subplots
```python
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2)
fig.add_scatter(y=[4, 2, 3.5], mode="markers", marker=dict(size=20, color="LightSeaGreen"), name="a", row=1, col=1)
fig.add_bar(y=[2, 1, 3], marker=dict(color="MediumPurple"), name="b", row=1, col=1)
fig.add_scatter(y=[2, 3.5, 4], mode="markers", marker=dict(size=20, color="MediumPurple"), name="c", row=1, col=2)
fig.add_bar(y=[1, 3, 2], marker=dict(color="LightSeaGreen"), name="d", row=1, col=2)
fig.update_traces(marker_color="RoyalBlue", selector=dict(marker_color="MediumPurple"))
fig.show()
```
![image](https://user-images.githubusercontent.com/56889151/97860101-ead8f200-1d44-11eb-955e-4283a4eabecc.png)

### 3D plot
```python
import plotly.graph_objects as go
import numpy as np

y, x = np.mgrid[-3:3:100j,-3:3:200j]
f = lambda x,y : 10*np.exp(-x**2-y**2) + np.random.normal(size=(100, 200))
fig = go.Figure(data=go.Surface(z=f(x,y)))
fig.show()
```
![image](https://user-images.githubusercontent.com/52376448/97774622-cf32e780-1b9c-11eb-8a2f-6a7fffe36a6a.png)

### Mapbox
```python
import pandas as pd
import plotly.express as px

korea = pd.read_csv("https://raw.githubusercontent.com/ailever/openapi/master/csv/korea.csv")
fig = px.scatter_mapbox(korea, lat="latitude", lon="longitude", hover_name="landmark", hover_data=["city"],
                        color_discrete_sequence=["fuchsia"], zoom=3, height=300)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()
```
![image](https://user-images.githubusercontent.com/52376448/97773142-aefd2b80-1b90-11eb-91a4-f8ca906c7c74.png)
