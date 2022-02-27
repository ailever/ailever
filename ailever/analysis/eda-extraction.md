
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
`pivot`
```sql
select 
        education
      , sum(case when trim(race) = 'White' then 1 else 0 end) as White
      , sum(case when trim(race) = 'Black' then 1 else 0 end) as Black
      , sum(case when trim(race) = 'Asian-Pac-Islander' then 1 else 0 end) as API
      , sum(case when trim(race) = 'Amer-Indian-Eskimo' then 1 else 0 end) as AIE   
      , sum(case when trim(race) = 'Other' then 1 else 0 end) as Other      
from adult
group by education
```
![image](https://user-images.githubusercontent.com/56889151/155875770-ff9cb3bd-386e-454d-a20f-74b2cab23662.png)



