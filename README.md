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


<!--
**ailever/ailever** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->


## Documentation
```
- docs 
  |-- build
  |-- source
      |-- conf.py
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
html_theme = 'sphinx_rtd_theme'
```
`ailever/docs`
```bash
$ make html
```


## Contact
- Email : ailever.group@gmail.com
