# APP
```
- __init__.py
- APPManagement.py
- AppFile.py

- AppEYES
- EYESLayout.py
- EYESApp.py

- AppBRAIN
- BRAINLayout.py
- BRAINApp.py
```

## Eyes Project
```
- EYESLayout.py
- EYESApp.py
- AppEYES
   |-- __init__.py
```
```python
from ailever.apps import eyes
eyes.download()
eyes.run(host='127.0.0.1', port='8050')
```

## Brain Project
```
- BRAINLayout.py
- BRAINApp.py
- AppBRAIN
   |-- __init__.py
   |-- P1T1.py
   |-- P1T2.py
```
```python
from ailever.apps import brain
brain.download()
brain.run(host='127.0.0.1', port='8050')
```

<br><br><br>
<hr>

## Apps Utils
### with dash
`directory`
```
- app.py
```
`app.py`
```python
import dash
import dash_core_components as dcc
import dash_html_components as html

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)
app.layout = html.Div(children=[html.H1(children='Hello Dash'),
				html.Div(children='Dash: A web application framework for Python.'),
    				dcc.Markdown('## Ailever!')])

if __name__ == '__main__':
    app.run_server(host="127.0.0.1", port='8050', debug=True)
```
```bash
$ python app.py
```

### with PyQt5
`directory`
```
- ailever.ui
- UIAilever.py
- app.py
```
`ailever.ui > UIAilever.py`
```bash
$ pyuic5 ailever.ui -o UIAilever.py
```
`app.py`
```python
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
from UIAilever import Ui_MainWindow

class AileverApp(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(AileverApp, self).__init__(parent)
        self.setupUi(self)

def run():
    app = QApplication(sys.argv)
    form = AileverApp()
    form.show()
    app.exec_()

if __name__ == '__main__':
    run()
```
```bash
$ python app.py
```
