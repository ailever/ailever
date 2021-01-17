import os
import signal
from multiprocessing import Queue, Process
import time

queue = Queue()

def remover(name, sustain=20, queue=queue):
    time.sleep(sustain)
    os.remove(f'./{name}.py')
    while not queue.empty():
        os.kill(queue.get(), signal.SIGTERM)

def _dashboard(name, host='127.0.0.1', port='8050', queue=queue):
    queue.put(os.getpid())
    os.system(f'python {name}.py --ds {host} --dp {port}')

def dashboard(name, host='127.0.0.1', port='8050', sustain=20, queue=queue):
    proc1 = Process(target=remover, args=(name, sustain, queue, ))
    proc2 = Process(target=_dashboard, args=(name, host, port, queue, ))
    proc1.start()
    proc2.start()
    proc1.join()
    proc2.join()
