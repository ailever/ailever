
## MySQL
### Table Extraction
```python
import pandas as pd
import pymysql

connection = pymysql.connect(host='localhost', user='[user_id]', password='[password]', db='[database]', charset='utf8')

query = """
select * from adult
"""

table = pd.read_sql_query(query, connection); connection.close()
table
```
![image](https://user-images.githubusercontent.com/56889151/155865692-981285c1-553c-46eb-9ea4-7fcd6204c6de.png)


### Grouping
#### COUNT() OVER()
```sql
select 
      sex
    , race
    , count(1)
    , count(1) over()
    , count(1) over(order by sex asc)
    , count(1) over(order by race asc)
    , count(1) over(partition by sex)
    , count(1) over(partition by sex order by race)    
    , count(1) over(partition by race)
    , count(1) over(partition by race order by sex)
from adult
group by 1, 2
order by 1, 2
```
![image](https://user-images.githubusercontent.com/56889151/155866046-e277a237-a3c3-4f1f-8115-5cf99e74c8be.png)




