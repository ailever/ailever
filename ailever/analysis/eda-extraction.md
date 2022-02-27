
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
#### COUNT OVER()
`(ROW_NUMBER) Ordinal Number` vs `(COUNT) Cardinal Number`
```sql
select 
      sex
    , race
    , row_number() over(order by sex, race)
    , count(1)
    ,     count(1) over()
    ,     count(1) over(order by sex)
    ,     count(1) over(partition by sex)
    ,     count(1) over(partition by sex order by race)    
    , row_number() over(partition by sex order by race)
    ,     count(1) over(order by race)
    ,     count(1) over(partition by race)
    ,     count(1) over(partition by race order by sex)
    , row_number() over(partition by race order by sex)
from adult
group by 1, 2
order by 1, 2
```
![image](https://user-images.githubusercontent.com/56889151/155866399-b4430438-97ec-4dcd-96a5-ea68499f054e.png)


```sql
select 
      sex
    , race
    , 50K
    , row_number() over(order by sex, race)
    , count(1)
    ,     count(1) over()
    ,     count(1) over(order by sex)
    ,     count(1) over(partition by sex)
    ,     count(1) over(partition by sex order by race)    
    , row_number() over(partition by sex order by race)
    ,     count(1) over(order by race)
    ,     count(1) over(partition by race)
    ,     count(1) over(partition by race order by sex)
    , row_number() over(partition by race order by sex)
from adult
group by 1, 2, 3
order by 1, 2, 3
```
![image](https://user-images.githubusercontent.com/56889151/155867279-149445d0-f74a-47c1-9f62-bb2dce5fa213.png)


### SUM OVER()
```sql
select 
      sex
    , race
    , count(0)    
    , count(1)
    , count(2)
    , sum(0)
    , sum(1)
    , sum(2)
    , count(0) over()
    , count(1) over()
    , count(2) over()
    , sum(0) over() 
    , sum(1) over()
    , sum(2) over()
    , count(count(1)) over()
    , count(sum(1)) over()
    , sum(sum(1)) over()
    , sum(count(1)) over()
from adult
group by 1, 2
order by 1, 2
```
![image](https://user-images.githubusercontent.com/56889151/155869554-1cc82776-9c86-4e9f-8397-7d101103770e.png)

