import os
from urllib.request import urlretrieve

def dashboard(name='main', host='127.0.0.1', port='8050'):
    if name == 'main':
        if not os.path.isfile(f'{name}.py'):
            urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/'+name+'.py', f'./{name}.py')
            print(f'[AILEVER] The file "{name}.py" is downloaded!')
    elif name[:2] == 'WS':
        if not os.path.isfile(f'{name}.py'):
            urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/work-sheet/'+name+'.py', f'./{name}.py')
            print(f'[AILEVER] The file "{name}.py" is downloaded!')
    elif name[:4] == 'PROJ':
        if not os.path.isfile(f'{name}.py'):
            urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/project/'+name+'.py', f'./{name}.py')
            print(f'[AILEVER] The file "{name}.py" is downloaded!')
    else:
        print(f'[AILEVER] Download is failed.')

    os.system(f'python {name}.py --ds {host} --dp {port}')
