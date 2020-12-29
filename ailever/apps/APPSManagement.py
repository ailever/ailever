from urllib.request import urlretrieve

class Eyes():
    def download(self):
        urlretrieve('https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/app_eyes.py', f'./app_eyes.py')
        print(f'[AILEVER] The file "app_eyes.py" is downloaded!')

    def run(self):
        from .app_eyes import eyes
        eyes.run()

class Brain():
    def download(self):
        urlretrieve('https://raw.githubusercontent.com/ailever/ailever/master/ailever/apps/app_brain.py', f'./app_brain.py')
        print(f'[AILEVER] The file "app_brain.py" is downloaded!')

    def run(self):
        from .app_brain import brain
        brain.run()

eyes = Eyes()
brain = Brain()
