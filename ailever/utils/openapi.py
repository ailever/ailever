import os
from urllib.request import urlretrieve

def source(name):
    urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/source/'+name+'.py', f'./{name}.py')
    print(f'[AILEVER] The file "{name}.py" is downloaded!')
