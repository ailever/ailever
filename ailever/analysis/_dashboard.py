import os
from urllib.request import urlretrieve

def dashboard(name, host='127.0.0.1', port='8050'):
    if not os.path.isfile(f'{name}.py'):
        urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/analysis/'+name+'.py', f'./{name}.py')
        print(f'[AILEVER] The file "{name}.py" is downloaded!')
    os.system(f'python {name}.py --ds {host} --dp {port}')

