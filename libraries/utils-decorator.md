## Function-Decorators
```python
```

## Class-Decorators
```python
class Trace:
    def __init__(self, func):
        self.func = func
    
    def __call__(self, a, b):
        print(self.func.__name__, '-- START --')
        self.func(a, b)
        print(self.func.__name__, '-- END --')

    def deco_method01(self):
        print('my_func method01')

    def deco_method02(self):
        print('my_func method02')
        
@Trace
def my_func(a, b):
    print('my_func core:', a + b)
        
my_func(a=1,b=2)
my_func.deco_method01()
my_func.deco_method02()
```


## Decorators inside a class
```python
```


## Inheritance of decorators inside a class
```python
```




