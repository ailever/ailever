```python
%%writefile test.py
with open('test.txt', 'w') as f:
    f.write('test')
```
```python
import subprocess
server = subprocess.Popen(["python", "test.py"])
```
```python
with open('test.txt', 'r') as f:
    file = f.read()
print(file)
```
