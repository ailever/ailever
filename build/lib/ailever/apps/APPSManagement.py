from urllib.request import urlretrieve

class Eyes():
    def download(self):
        name = 'app_eyes.py'
        urlretrieve(f'https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/{name}', f'./{name}')
        print(f'[AILEVER] The file "{name}" is downloaded!')

    def run(self):
        from .app_eyes import eyes
        eyes.run()

class Brain():
    def download(self):
        name = 'app_brain.py'
        urlretrieve(f'https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/{name}', f'./{name}')
        print(f'[AILEVER] The file "{name}" is downloaded!')

        name = 'P1T1.py'
        urlretrieve(f'https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/AppBRAIN/{name}', f'./{name}')
        print(f'[AILEVER] The file "{name}" is downloaded!')

        name = 'P1T2.py'
        urlretrieve(f'https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/AppBRAIN/{name}', f'./{name}')
        print(f'[AILEVER] The file "{name}" is downloaded!')

    def run(self):
        from .app_brain import brain
        brain.run()

eyes = Eyes()
brain = Brain()
