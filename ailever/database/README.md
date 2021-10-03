```python
from ailever.database import DB

db = DB('postgresql', verbose=False)
db.connection(user='id', password='passwd')
db.execute("""
SELECT * FROM COLS;
""")
```
