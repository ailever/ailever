```python
from ailever.databases import DB

db = DB('postgresql')
db.connection(user='id', password='passwd')
db.execute("""
SELECT * FROM COLS;
""")
```
