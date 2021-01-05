## Ailever｜[Homepage](https://ailever.github.io/)
- https://github.com/ailever/ailever.github.io
- https://github.com/ailever/ailever
- https://github.com/ailever/programming-language
- https://github.com/ailever/numerical-method
- https://github.com/ailever/statistics
- https://github.com/ailever/machine-learning
- https://github.com/ailever/deep-learning
- https://github.com/ailever/reinforcement-learning
- https://github.com/ailever/applications
- https://github.com/ailever/security
- https://github.com/ailever/openapi

<br><br><br>

***

## Setup
```bash
# pip freeze > requirements.txt
$ pip install -r requirements.txt
```

`ailever/ailever/`
```bash
$ bash update.sh
```
`ailever/`
```bash
$ python setup.py bdist_wheel
$ twine upload dist/ailever-0.0.1-py3-none-any.whl      # (or) python -m twine upload dist/ailever-0.0.1-py3-none-any.whl
```


<br><br><br>

***

## Documentation
[hosting](https://readthedocs.org/)
```
- docs
  |-- requirements.txt
  |-- build
  |-- source
      |-- conf.py
      |-- index.rst
      |-- introduction.rst
      |-- installation.rst
      |-- eyes
          |-- index.rst
      |-- brain
          |-- index.rst
      |-- machine
          |-- index.rst
          |-- RL
          |-- DL
          |-- ML
          |-- ST
          |-- NM
      |-- language
          |-- index.rst
      |-- captioning
          |-- index.rst
      |-- detection
          |-- index.rst
      |-- forecast
          |-- index.rst
  |--  make.bat
  |--  Makefile
```
```bash
$ pip install sphinx_rtd_theme
$ pip install recommonmark
```
`ailever/docs/source/conf.py`
```python
import os
import sys
sys.path.insert(0, os.path.abspath("../../ailever/"))

extensions = ['recommonmark', 'sphinx.ext.autodoc']
html_theme = 'sphinx_rtd_theme'
```
`ailever/docs`
```bash
$ pip list --format=freeze > requirements.txt
$ sphinx-apidoc -f -o ./source/machine ../ailever/machine
$ make clean
$ make html
```


<br><br><br>

***

## Contact
- Email : ailever.group@gmail.com
