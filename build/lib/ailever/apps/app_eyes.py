from urllib.request import urlretrieve
import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
app.layout = html.Div(
        children=[
            html.H1(children='Hello Dash'),
            html.Div(children='Dash: A web application framework for Python.'),
            dcc.Markdown('## Ailever!')])


Eyes = type('Eyes', (), {})
eyes = Eyes()
eyes.donwload = lambda : urlretrieve('https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/app_eyes.py', f'./app_eyes.py')
eyes.run = lambda : app.run_server(host="127.0.0.1", port='8050', debug=True)

if __name__ == "__main__":
    eyes.run()
