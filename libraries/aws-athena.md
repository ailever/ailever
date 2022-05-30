## [Cloud Computing] | [Docs]() | [GitHub]() | [PyPI](https://pypi.org/project/pyathena/)

- https://wikidocs.net/book/2203


```python
import boto3
from pyathena import connect
import pandas as pd

sess = boto3.Session()
region = sess.region_name

conn = connect(s3_staging_dir="s3://ailever-athena/dataset", region_name=region)
pd.read_sql_query("SELECT * FROM table", conn)
```
