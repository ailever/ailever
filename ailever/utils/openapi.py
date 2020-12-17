import os
from urllib.request import urlretrieve

def source(name):
    urlretrieve('https://github.com/ailever/openapi/raw/master/source/'+name, f'./{name}.py')
    print(f'[AILEVER] The file "{name}.py" is downloaded!')
