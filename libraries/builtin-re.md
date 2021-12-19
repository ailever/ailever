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

print(re.findall('a\s\s', 'abc')) # []
print(re.findall('a\s\s', 'a  ')) # ['a  ']
print(re.findall('a\s\s', 'a\t ')) # ['a\t ']
print(re.findall('a\s\s', 'a\n ')) # ['a\n ']
print(re.findall('a\s\s', 'a \t')) # ['a \t']
print(re.findall('a\s\s', 'a \n')) # ['a \n']
print(re.findall('a\s\s', 'a\t\n')) # ['a\t\n']
print(re.findall('a\s\s', 'a\n\t')) # ['a\n\t']
print(re.findall('a\s\s', 'a\t\t')) # ['a\t\t']
print(re.findall('a\s\s', 'a\n\n')) # ['a\n\n']
```
```python
import re

print(re.findall(' abc\s',  ' abc \tabc\t')) # [' abc ']
print(re.findall('\sabc ',  ' abc \tabc\t')) # [' abc ']
print(re.findall(' abc ',   ' abc \tabc\t')) # [' abc ']
print(re.findall('\sabc\s', ' abc \tabc\t')) # [' abc ', '\tabc\t']
```
```python
import re

print(re.findall('^ abc\d\s$', ' abc1 '))                                     # [' abc1 ']
print(re.findall('^ abc\d\s$', ' abc1 \t'))                                   # []
print(re.findall('^ abc\d\s$', ' abc1 \n'))                                   # [' abc1 ']
print(re.findall('^ abc\d\s$', ' abc1 \n abc2 \n abc3 \n abc4 '))             # []
print(re.findall('^ abc\d\s$', ' abc1 \n abc2 \n abc3 \n abc4 ', re.M))       # [' abc1 ', ' abc2 ', ' abc3 ', ' abc4 ']
```

### Re metaword
`\w, \W`
```python
import re

print(re.findall('\w',  'abc'))     # ['a', 'b', 'c']
print(re.findall('\w',  '...'))     # []
print(re.findall('\w',  '\\\\\\'))  # []
print(re.findall('\W', 'abc'))      # []
print(re.findall('\W',  '...'))     # ['.', '.', '.']
print(re.findall('\W',  '\\\\\\'))  # ['\\', '\\', '\\']
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
re.search("a", text)        # match:a, span:(3,4)
re.search("a", text, re.I)  # match:A, span:(0,1)
```

#### S: .(dot) on pattern <-> end of line(\n) on text

```python
import re

print(re.findall('a..', 'abc a  a\n\n'))         # ['abc', 'a  ']
print(re.findall('a..', 'abc a  a\n\n', re.S))   # ['abc', 'a  ', 'a\n\n']

print(re.findall('a..', 'abc a  a\\\\'))         # ['abc', 'a  ', 'a\\\\']
print(re.findall('a..', 'abc a  a\\\\', re.S))   # ['abc', 'a  ', 'a\\\\']
```




