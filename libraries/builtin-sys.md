```python
import sys

def main_function():
    sub_function()

def sub_function():
    print(sys._getframe(0).f_code.co_name)
    print(sys._getframe(1).f_code.co_name)
```
