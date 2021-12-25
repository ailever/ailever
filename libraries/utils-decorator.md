## Function-Decorators
```python
def Trace(func):
    def wrapper(*args, **kwargs): # args: (1, 2), kwargs: {}
        return func(*args, **kwargs) 
    return wrapper

@Trace
def my_func(a, b):
    print(a + b)

my_func(1,2)    
```
```python
def Trace(*args, **kwargs): # args: (3, 4), kwargs: {}
    def decorator(func):
        def wrapper(*args, **kwargs): # args: (1, 2), kwargs: {}
            return func(*args, **kwargs) 
        return wrapper
    return decorator

@Trace(3, 4)
def my_func(a, b):
    print(a + b)

my_func(1,2)
```


## Class-Decorators
```python
class Trace:
    def __init__(self, func): # args: (3, 4), kwargs: {}
        self.func = func
    
    def __call__(self, *args, **kwargs): # args: (1, 2), kwargs: {}
        self.func(*args, **kwargs) 

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

```python
class wrapper:
    def __init__(self, func, *args, **kwargs): # args: (3, 4), kwargs: {}
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):  # args: (1, 2), kwargs: {}
        self.func(*args, **kwargs)

def Trace(*args, **kwargs): # args: (3, 4), kwargs: {}
    def decorator(func):
        return wrapper(func, *args, **kwargs) 
    return decorator

@Trace(3, 4)
def my_func(a, b):
    print(a + b)

my_func(1, 2)
```

```python
class Trace:
    def __init__(self, *args, **kwargs): # args: (3, 4), kwargs: {}
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, func):
        def wrapper(*args, **kwargs): # args: (1, 2), kwargs: {}
            return func(*args, **kwargs) 
        return wrapper
        
@Trace(3,4)
def my_func(a, b):
    print(a + b)
        
my_func(1,2)
```

## Decorators inside a class
```python

```


## Inheritance of decorators inside a class
```python
```




