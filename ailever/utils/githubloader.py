import os

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
    elif repo == 'applications':
        os.system('git clone https://github.com/ailever/applications.git')
        print(f'[AILEVER] The repository "{repo}" is successfully cloned!')
    elif repo == 'deep-learning':
        os.system('git clone https://github.com/ailever/deep-learning.git')
        print(f'[AILEVER] The repository "{repo}" is successfully cloned!')

