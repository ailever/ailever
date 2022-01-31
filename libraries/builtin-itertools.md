
## CustomClass
```python
import itertools
from collections import defaultdict

class CustomClass:
    _class_counter1 = itertools.count() # 0
    _class_counter2 = itertools.count() # 0
    _class_counter3 = itertools.count() # 0
    _class_counter_dict = defaultdict(itertools.count) # 0
    
    def __new__(cls):
        print('[CLS]', cls._class_counter1)
        return next(cls._class_counter1)
```
```python
custom_obj = CustomClass()
print(CustomClass._class_counter1)
print(CustomClass._class_counter2)
print(next(CustomClass._class_counter3))
print(CustomClass._class_counter_dict[0])
print(next(CustomClass._class_counter_dict[1]))
custom_obj
```
