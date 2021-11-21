# Ailever｜[Homepage](https://ailever.github.io/)
- https://github.com/ailever/ailever.github.io
- https://github.com/ailever/ailever
- https://github.com/ailever/project-templates
- https://github.com/ailever/dataset
- https://github.com/ailever/analysis
- https://github.com/ailever/forecast
- https://github.com/ailever/investment
- https://github.com/ailever/programming-language
- https://github.com/ailever/numerical-method
- https://github.com/ailever/statistics
- https://github.com/ailever/machine-learning
- https://github.com/ailever/deep-learning
- https://github.com/ailever/reinforcement-learning
- https://github.com/ailever/applications
- https://github.com/ailever/security
- https://github.com/ailever/openapi

<br><br><br><br><br>

***

## Setup
```bash
# pip freeze > requirements.txt
$ pip install -r requirements.txt
```
`ailever/`
```bash
$ bash distribution.sh [version]
```


<br><br><br><br><br>

***

## Documentation｜[Docs](https://ailever.readthedocs.io/en/latest/)
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
          |-- machine.RL.rst
          |-- machine.DL.rst
          |-- machine.ML.rst
          |-- machine.ST.rst
          |-- machine.NM.rst
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
`ailever/docs/source/conf.py`
```python
import os
import sys
sys.path.insert(0, os.path.abspath("../../ailever/"))

extensions = ["sphinx.ext.autodoc",
    	        "sphinx.ext.coverage",
    	        "sphinx.ext.mathjax",
    	        "sphinx.ext.viewcode",
    	        "sphinx.ext.autosectionlabel",
    	        "sphinx.ext.napoleon"]
html_theme = 'sphinx_rtd_theme'
```
`ailever/.readthedocs.yml`
```
version: 2

sphinx:
  configuration: docs/source/conf.py

python:
  version: 3.6
  install:
    - requirements: docs/requirements.txt
```
`ailever/docs`
```bash
$ pip install sphinx_rtd_theme
$ pip list --format=freeze > requirements.txt
```
```bash
$ sphinx-apidoc -f -o ./source/machine ../ailever/machine/RL
$ sphinx-apidoc -f -o ./source/machine ../ailever/machine/DL
$ sphinx-apidoc -f -o ./source/machine ../ailever/machine/ML
$ sphinx-apidoc -f -o ./source/machine ../ailever/machine/ST
$ sphinx-apidoc -f -o ./source/machine ../ailever/machine/NM
$ sphinx-apidoc -f -o ./source/forecast ../ailever/forecast/STOCK
$ sphinx-apidoc -f -o ./source/utils ../ailever/utils
```
```bash
$ make clean
$ make html
```

<br><br><br><br><br>

***

## Dependancy
```bash
$ sudo apt install python3-pipdeptree
$ pip install pipdeptree
$ pipdeptree
```
```bash
$ pip install --no-index --find-links [wheelfile]
```

<br><br><br><br><br>

***

## Contact
- Email : ailever.group@gmail.com


<br><br><br><br><br>

***

