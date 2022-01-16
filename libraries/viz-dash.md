## [Web]|[dash](https://dash.plotly.com/) | [github](https://github.com/plotly/dash)

## Port Check
```bash
$ netstat -nlpt
$ kill -9 [PID]
```

## Dash Basic
`app.py`
```python
# visit http://127.0.0.1:8050/ in your web browser.
import dash
from dash import dcc
from dash import html

app = dash.Dash(__name__)
app.layout = dcc.Markdown("""
## Hello, World
""")

if __name__ == '__main__':
    app.run_server(host="127.0.0.1", port='8050', debug=True)
```
```bash
$ python app.py
```

## Dash Image
`current path`
```python
import dash
from dash import dcc
from dash import html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
img_path = app.get_asset_url('test.png')

app.layout = dcc.Markdown(f"""
![my_image]({img_path})
""")

if __name__ == '__main__':
    app.run_server(host="127.0.0.1", port='8050', debug=True)
```
`relative path`
```python
import dash
from dash import dcc
from dash import html
import os

app = dash.Dash(__name__, assets_folder=os.getcwd() + '/img') # file path = ./img/test.png 
dash_img_path = app.get_asset_url('test.png')                 # dash_img_path = /assets/test.png
app.layout = dcc.Markdown(f"""
![my_image]({dash_img_path})
""")

if __name__ == '__main__':
    app.run_server(host="127.0.0.1", port='8050', debug=True)
```
