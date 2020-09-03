import os
from urllib.request import urlretrieve

def storage(name):
    if name == 'list':
        print('[AILEVER] List of contents in the storage')
        contents = ['experiment.tar.gz', 'experiment.zip',
                    'template.tar.gz', 'template.zip']
        for content in contents:
            print(f'* {content}')
    else:
        urlretrieve('https://raw.githubusercontent.com/ailever/ailever/master/storage/'+name, f'./{name}')

    if name != 'list':
        print(f'[AILEVER] The file "{name}" is downloaded!')


def repository(repo):
    if repo == 'list':
        print('[AILEVER] List of contents in the repository')
        contents = ['ailever',
                    'programming-language',
                    'numerical-method',
                    'statistics',
                    'applications',
                    'deep-learning']
        for content in contents:
            print(f'* {content}')
    elif repo == 'ailever':
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
    
    if repo != 'list':
        print(f'[AILEVER] The repository "{repo}" is successfully cloned!')


def cloud(name=None):
    if name == 'list':
        print('[AILEVER] List of contents in the cloud')
        contents = ['template.pptx',
                    'regression.pptx']
        for content in contents:
            print(f'* {content}')

    elif name == 'template.pptx':
        urlretrieve('https://docs.google.com/uc?export=download&id=1MFQuoE1B58TSice5aPDqQDdSJqu1CLB3', f'./{name}')
    elif name == 'regression.pptx':
        urlretrieve('https://docs.google.com/uc?export=download&id=1OsS7Dd56yf9VftF7BBLWxECI9l_xmksG', f'./{name}')
    
    if name != 'list':
        print(f'[AILEVER] The file "{name}" is downloaded!')


