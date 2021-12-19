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

### Spacing word
```python
import re

print(re.findall('a\s\s', 'abc a  a\t\n')) # ['a  ', 'a\t\n']
print(re.findall('a\s\s', 'abc a  a\n\t')) # ['a  ', 'a\n\t']
print(re.findall('a\s\s', 'abc a  a\t\t')) # ['a  ', 'a\t\t']
print(re.findall('a\s\s', 'abc a  a\n\n')) # ['a  ', 'a\n\n']
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

#### S: .(dot) on pattern <-> end of line(\n) on text

```python
import re

print(re.findall('a..', 'abc a  a\n\n'))         # ['abc', 'a  ']
print(re.findall('a..', 'abc a  a\n\n', re.S))   # ['abc', 'a  ', 'a\n\n']

print(re.findall('a..', 'abc a  a\\\\'))         # ['abc', 'a  ', 'a\\\\']
print(re.findall('a..', 'abc a  a\\\\', re.S))   # ['abc', 'a  ', 'a\\\\']
```




