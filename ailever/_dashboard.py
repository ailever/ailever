import os

def dashboard(name, host='127.0.0.1', port='8050'):
    os.system(f'python {name}.py --ds {host} --dp {port}')
