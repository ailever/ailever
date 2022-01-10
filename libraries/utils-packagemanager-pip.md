## [Package Management] | [pip](https://pypi.org/project/pip/) | [github](https://github.com/pypa/pip)

```bash
$ pip list
$ pip freeze
$ pip show [package]
$ pip download -d [path] [package]
```

## Installation
`Online`
```bash
$ pip install [package]
$ pip freeze > requirements.txt
$ pip install -r requirements.txt
```

`Offline`
```bash
$ pip install [package] --no-index --find-links [path]
$ pip install --no-cache-dir [package] --no-index --find-links [path]
```

## Dependencies
```bash
$ pip show [package]
```
```bash
$ sudo apt install python3-pipdeptree
$ pip install pipdeptree
$ pipdeptree -rp [package]
```


## Packages
### Pygments
```bash
$ pip install Pygments
```
```bash
$ pygmentize file.py
```
