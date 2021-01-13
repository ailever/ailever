import os
from urllib.request import urlretrieve

def dashboard(name):
    urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/language/'+name+'.py', f'./{name}.py')
    print(f'[AILEVER] The file "{name}.py" is downloaded!')

