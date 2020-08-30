import os
from urllib.request import urlretrieve

def storage(name):
    urlretrieve('https://raw.githubusercontent.com/ailever/ailever/master/storage/'+name, f'./{name}')
    print(f'[AILEVER] The file "{name}" is downloaded!')


def repository(repo):
    if repo == 'ailever':
        os.system('git clone https://github.com/ailever/ailever.git')
    elif repo == 'programming-language':
        os.system('git clone https://github.com/ailever/programming-language.git')
    elif repo == 'numerical-method':
        os.system('git clone https://github.com/ailever/numerical-method.git')
    elif repo == 'statistics':
        os.system('git clone https://github.com/ailever/statistics.git')
    elif repo == 'applications':
        os.system('git clone https://github.com/ailever/applications.git')
    elif repo == 'deep-learning':
        os.system('git clone https://github.com/ailever/deep-learning.git')
    
    print(f'[AILEVER] The repository "{repo}" is successfully cloned!')


def cloud(name=None):
    if name == 'template.pptx':
        urlretrieve('https://docs.google.com/uc?export=download&id=1MFQuoE1B58TSice5aPDqQDdSJqu1CLB3', f'./{name}')
    elif name == 'regression.pptx':
        urlretrieve('https://docs.google.com/uc?export=download&id=1OsS7Dd56yf9VftF7BBLWxECI9l_xmksG', f'./{name}')

    print(f'[AILEVER] The file "{name}" is downloaded!')


