
```python
import numpy as np
from multiprocessing import Process, Queue

def my_func(a, b, queue):
    global A
    A = a+b
    queue.put(A)

A = 1
queue = Queue()
procs = []
for i in range(5):
    a, b = np.random.normal(size=2)
    proc = Process(target=my_func, args=(a, b, queue, ))
    procs.append(proc)
    proc.start()
    
for proc in procs:
    proc.join()

ipc_message_queue = list()    
for _ in range(queue.qsize()):
    ipc_message_queue.append(queue.get())

ipc_message_queue, A
```
