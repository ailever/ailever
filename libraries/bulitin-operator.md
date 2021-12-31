
```python
from operator import attrgetter

class CLS:
    def __init__(self):
        self.property = 1
                
obj = CLS()
attrgetter('property')(obj)
```
