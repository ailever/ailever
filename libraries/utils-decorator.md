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
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        self.func(*args, **kwargs)

    def deco_method01(self):
        print('my_func method01')

    def deco_method02(self):
        print('my_func method02')

def Trace(*args, **kwargs):
    def decorator(func):
        return wrapper(func, *args, **kwargs)
    return decorator

@Trace(3, 4)
def my_func(a, b):
    print(a + b)

my_func(1, 2)
my_func.deco_method01()
my_func.deco_method02()
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
class MyClass:
    def Trace(func) :
        def wrapper(self, *args, **kwargs): # args: (1, 2), kwargs: {}
            return func(self, *args, **kwargs)
        return wrapper
  
    @Trace
    def my_func(self, a, b) :
        print('Decorating - MyClass methods.')
    
my_obj = MyClass()
my_obj.my_func(1, 2)
```

```python
class MyClass:
    def Trace(*args, **kwargs): # args: (3, 4), kwargs: {}
        def decorator(func) :
            def wrapper(self, *args, **kwargs): # args: (1, 2), kwargs: {}
                return func(self, *args, **kwargs)
            return wrapper
        return decorator
  
    @Trace(3,4)
    def my_func(self, a, b) :
        print('Decorating - MyClass methods.')
    
my_obj = MyClass()
my_obj.my_func(1, 2)
```


## Inherientance of decorators inside a class
```python
class MyParentClass:
    def Trace(func) :
        def wrapper(self) :
            func(self)
        return wrapper
  
    @Trace
    def my_parent_func(self) :
        print('Decorating - MyParentClass methods.')
  
class MyClass(MyParentClass):
    @MyParentClass.Trace
    def my_func(self) :
        print('Decoration - MyClass methods.')
  
my_obj = MyClass()
my_obj.my_func()
my_obj.my_parent_func()
```




