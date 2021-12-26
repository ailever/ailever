## Function-Decorators
```python
from functools import wraps

def Trace(func):
    @wraps(func)
    def wrapper(*args, **kwargs): # args: (1, 2), kwargs: {}
        return func(*args, **kwargs) 
    return wrapper

@Trace
def my_func(a, b):
    print(a + b)

print(my_func.__name__)
my_func(1,2)    
```
```python
from functools import wraps

def Trace(*args, **kwargs): # args: (3, 4), kwargs: {}
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs): # args: (1, 2), kwargs: {}
            return func(*args, **kwargs) 
        return wrapper
    return decorator

@Trace(3, 4)
def my_func(a, b):
    print(a + b)

print(my_func.__name__)
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
        
print(my_func)
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

print(my_func)
my_func(1, 2)
my_func.deco_method01()
my_func.deco_method02()
```

```python
from functools import wraps

class Trace:
    def __init__(self, *args, **kwargs): # args: (3, 4), kwargs: {}
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self, func):
        @wraps(func)        
        def wrapper(*args, **kwargs): # args: (1, 2), kwargs: {}
            return func(*args, **kwargs) 
        return wrapper
        
@Trace(3,4)
def my_func(a, b):
    print(a + b)

print(my_func.__name__)    
my_func(1,2)
```

## Decorators inside a class
```python
from functools import wraps

class MyClass:
    def Trace(func) :
        @wraps(func)    
        def wrapper(self, *args, **kwargs): # args: (1, 2), kwargs: {}, self: MyClass <object>
            return func(self, *args, **kwargs)
        return wrapper
  
    @Trace
    def my_func(self, a, b) : # self: MyClass <object>
        print('Decorating - MyClass methods.')
    
my_obj = MyClass()
my_obj.my_func(1, 2)
print(my_obj.my_func.__name__)
```

```python
from functools import wraps

class MyClass:
    def Trace(*args, **kwargs): # args: (3, 4), kwargs: {}
        def decorator(func) :
            @wraps(func)    
            def wrapper(self, *args, **kwargs): # args: (1, 2), kwargs: {}, self: MyClass <object>
                return func(self, *args, **kwargs)
            return wrapper
        return decorator
  
    @Trace(3,4)
    def my_func(self, a, b) : # self: MyClass <object>
        print('Decorating - MyClass methods.')
    
my_obj = MyClass()
my_obj.my_func(1, 2)
print(my_obj.my_func.__name__)
```

```python
from functools import wraps

class MyClass:
    class Trace:
        def __init__(self, *args, **kwargs): # args: (3, 4), kwargs: {}, self: MyClass.Trace <object>
            self.args = args
            self.kwargs = kwargs

        def __call__(self, func): # self: MyClass.Trace <object>
            @wraps(func)            
            def wrapper(self, *args, **kwargs): # args: (1, 2), kwargs: {}, self: MyClass <object>
                return func(self, *args, **kwargs) 
            return wrapper
    
    @Trace(3,4)
    def my_func(self, a, b) :
        print('Decorating - MyClass methods.')

my_obj = MyClass()
my_obj.my_func(1, 2)        
print(my_obj.my_func.__name__)
```

```python
from functools import wraps

class MyClass:
    class Trace:
        def __init__(self, *args, **kwargs): # args: (3, 4), kwargs: {}, self: MyClass.Trace <object> 
            self.args = args
            self.kwargs = kwargs

        def __call__(self, func): # self: MyClass.Trace <object>
            @wraps(func)            
            def wrapper(cls, *args, **kwargs): # args: (1, 2), kwargs: {}, cls: MyClass <class>
                return func(cls, *args, **kwargs)
            return wrapper

    @classmethod
    @Trace(3,4)
    def my_func(cls, a, b) :
        print('Decorating - MyClass methods.')

my_obj = MyClass()
my_obj.my_func(1, 2)
print(my_obj.my_func.__name__)
```


## Inherientance of decorators inside a class
```python
from functools import wraps

class MyParentClass:
    def Trace(func) :
        @wraps(func)                    
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

print(my_obj.my_func)
print(my_obj.my_parent_func)
```




