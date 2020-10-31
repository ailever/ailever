## [Visualization]|[plotly](https://plotly.com/python/) | [github](https://github.com/plotly/plotly.py)

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
