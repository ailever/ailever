import os
from urllib.request import urlretrieve

def storage(name):
    urlretrieve('https://raw.githubusercontent.com/ailever/ailever/master/storage/'+name, f'./{name}')
    print(f'[AILEVER] The file "{name}" is downloaded!')


def repository(repo):
    if repo == 'ailever':
        os.system('git clone https://github.com/ailever/ailever.git')
        print(f'[AILEVER] The repository "{repo}" is successfully cloned!')
    elif repo == 'programming-language':
        os.system('git clone https://github.com/ailever/programming-language.git')
        print(f'[AILEVER] The repository "{repo}" is successfully cloned!')
    elif repo == 'numerical-method':
        os.system('git clone https://github.com/ailever/numerical-method.git')
        print(f'[AILEVER] The repository "{repo}" is successfully cloned!')
    elif repo == 'statistics':
        os.system('git clone https://github.com/ailever/statistics.git')
        print(f'[AILEVER] The repository "{repo}" is successfully cloned!')
    elif repo == 'applications':
        os.system('git clone https://github.com/ailever/applications.git')
        print(f'[AILEVER] The repository "{repo}" is successfully cloned!')
    elif repo == 'deep-learning':
        os.system('git clone https://github.com/ailever/deep-learning.git')
        print(f'[AILEVER] The repository "{repo}" is successfully cloned!')


def cloud(name=None):
    urlretrieve('https://docs.google.com/uc?export=download&id=1MFQuoE1B58TSice5aPDqQDdSJqu1CLB3', f'./template.pptx')
    print(f'[AILEVER] The file "template.pptx" is downloaded!')


