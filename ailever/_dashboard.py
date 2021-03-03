import os
from urllib.request import urlretrieve
import signal
from multiprocessing import Queue, Process
import time

queue = Queue()

def remover(name, sustain, queue=queue):
    if sustain:
        time.sleep(sustain)
        os.remove(f'./{name}.py')
        while not queue.empty():
            os.kill(queue.get(), signal.SIGTERM)

def _dashboard(name, host='127.0.0.1', port='8050', queue=queue):
    if not os.path.isfile(f'{name}.py'):
        urlretrieve('https://raw.githubusercontent.com/ailever/openapi/master/main.py', f'./main.py')
        print(f'[AILEVER] The file "{name}.py" is downloaded!')
    queue.put(os.getpid())
    os.system(f'python {name}.py --ds {host} --dp {port}')

def dashboard(name, host='127.0.0.1', port='8050', sustain=None, queue=queue):
    proc1 = Process(target=remover, args=(name, sustain, queue, ))
    proc2 = Process(target=_dashboard, args=(name, host, port, queue, ))
    proc1.start()
    proc2.start()
    proc1.join()
    proc2.join()
