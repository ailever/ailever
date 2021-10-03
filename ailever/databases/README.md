```python
from ailever.databases import DB

db = DB('postgresql')
db.connection(user='', password='')
db.execute("""
SELECT * FROM COLS;
""")
```
