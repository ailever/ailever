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

def eyes():
    app.run_server(host="127.0.0.1", port='8050', debug=True)


