## RE module
`RE functions`
```python
import re

re.match(pattern, text, flag)	
re.search(pattern, text, flag)	
re.findall(pattern, text, flag)	
re.finditer(pattern, text, flag)	
re.sub(pattern, text, flag)	
```

### Backslash
```python
import re

print(re.findall('a..', 'abc a  a\\'))        # ['abc', 'a  ']
print(re.findall('a..', 'abc a  a\\ '))       # ['abc', 'a  ', 'a\\ ']
print(re.findall('a..', 'abc a  a\\\\'))      # ['abc', 'a  ', 'a\\\\']
print(re.findall('a..', 'abc a  a\\\\ '))     # ['abc', 'a  ', 'a\\\\']
print(re.findall('a..', 'abc a  a\\\\\\'))    # ['abc', 'a  ', 'a\\\\']
print(re.findall('a..', 'abc a  a\\\\\\ '))   # ['abc', 'a  ', 'a\\\\']
```

### Re flags
`RE Flags`
```python
import re

re.I	#re.IGNORECASE	
re.M	#re.MULTILINE		
re.S	#re.DOTALL	
re.A	#re.ASCII	
re.U	#re.UNICODE
re.L	#re.LOCALE	
re.X	#re.VERBOSE	
```

#### I: Case
```python
import re

text = """ABCabc"""
re.search("a", text) # match:a, span:(3,4)
re.search("a", text, re.I) # match:A, span:(0,1)
```

#### S: End of Line

```python
import re

print(re.findall('a..', 'abc a  a\n\n'))
print(re.findall('a..', 'abc a  a\n\n', re.S))

print(re.findall('a..', 'abc a  a\\\\'))
print(re.findall('a..', 'abc a  a\\\\', re.S))
```
