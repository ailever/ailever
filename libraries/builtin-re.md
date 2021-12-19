### RE Flags
```python
re.I	#re.IGNORECASE	
re.M	#re.MULTILINE		
re.S	#re.DOTALL	
re.A	#re.ASCII	
re.U	#re.UNICODE
re.L	#re.LOCALE	
re.X	#re.VERBOSE	
```
```python
import re

text = """ABCabc"""
re.search("a", text) # match:a, span:(3,4)
re.search("a", text, re.I) # match:A, span:(0,1)
```
