import os
from urllib.request import urlretrieve

def dashboard(name='main', host='127.0.0.1', port='8050'):
    if name=='main':
        if not os.path.isfile(f'{name}.py'):
            urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/'+name+'.py', f'./{name}.py')
            print(f'[AILEVER] The file "{name}.py" is downloaded!')
    else:
        if not os.path.isfile(f'{name}.py'):
            urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/work-sheet/'+name+'.py', f'./{name}.py')
            print(f'[AILEVER] The file "{name}.py" is downloaded!')

    os.system(f'python {name}.py --ds {host} --dp {port}')
