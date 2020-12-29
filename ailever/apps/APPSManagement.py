from urllib.request import urlretrieve

class Eyes():
    def download(self):
        name = 'EYESLayout.py'
        urlretrieve(f'https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/{name}', f'./{name}')
        print(f'[AILEVER] The file "{name}" is downloaded!')

        name = 'EYESApp.py'
        urlretrieve(f'https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/{name}', f'./{name}')
        print(f'[AILEVER] The file "{name}" is downloaded!')

    def run(self):
        from .EYESLayout import eyes
        eyes.run()

class Brain():
    def download(self):
        name = 'BRAINLayout.py'
        urlretrieve(f'https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/{name}', f'./{name}')
        print(f'[AILEVER] The file "{name}" is downloaded!')

        name = 'BRAINApp.py'
        urlretrieve(f'https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/{name}', f'./{name}')
        print(f'[AILEVER] The file "{name}" is downloaded!')

        name = 'P1T1.py'
        urlretrieve(f'https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/AppBRAIN/{name}', f'./{name}')
        print(f'[AILEVER] The file "{name}" is downloaded!')

        name = 'P1T2.py'
        urlretrieve(f'https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/AppBRAIN/{name}', f'./{name}')
        print(f'[AILEVER] The file "{name}" is downloaded!')

    def run(self):
        from .BRAINLayout import brain
        brain.run()

eyes = Eyes()
brain = Brain()
