
## Logging
### Working Time
```python
from functools import wraps
from datetime import datetime
import pytz

def Trace(*args, **kwargs): # args: (3, 4), kwargs: {}
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs): # args: (1, 2), kwargs: {}
            start = datetime.now(pytz.timezone('Asia/Seoul'))            
            retval = func(*args, **kwargs)
            end = datetime.now(pytz.timezone('Asia/Seoul'))
            
            print('_'*70)
            print(f'* Function Name: {func.__name__}')
            print('%17s'%'- Working Time:', end - start)
            print('%17s'%'- Start Time:', start)
            print('%17s'%'- End Time:', end)
            print('_'*70)
            
            return retval
        return wrapper

    # Current Time
    return decorator

@Trace(3, 4)
def my_func(a, b):
    print(a + b)

my_func(1,2)
```

### 
```python

```
