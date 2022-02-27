
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




### Univariate Frequency Analysis
```sql
select 
      education
    , count(1)
    , sum(count(1)) over(partition by education) / sum(count(1)) over()
    , dense_rank() over(order by count(1) desc)
from adult
group by education
```
![image](https://user-images.githubusercontent.com/56889151/155881594-2b37d689-00ec-4944-b1e3-71cea643beee.png)


### Multivariate Frequency Analysis
```sql
select 
      education, relationship
    , count(1)
    , sum(count(1)) over(partition by education, relationship) / sum(count(1)) over(partition by education)
    , dense_rank() over(partition by education order by count(1))
from adult
group by education, relationship
```
![image](https://user-images.githubusercontent.com/56889151/155881602-9d6ed440-c4c6-40bc-9ea8-acf109d334c6.png)




### Hierarchical Frequency Analysis
```sql
select 
      education
    , sum(count(1)) over(partition by education)
    , sum(count(1)) over(partition by education) / sum(count(1)) over()
    , relationship
    , sum(count(1)) over(partition by education, relationship)
    , sum(count(1)) over(partition by education, relationship) / sum(count(1)) over(partition by education)
    , race
    , sum(count(1)) over(partition by education, relationship, race)
    , sum(count(1)) over(partition by education, relationship, race) / sum(count(1)) over(partition by education, relationship)
from adult
group by education, relationship, race
order by education, relationship, race
```
![image](https://user-images.githubusercontent.com/56889151/155878488-14e01e04-5b2d-490f-aa60-61748d7b0e20.png)

```sql
select 
      education
    , HC_Count
    , HC_Ratio
    , dense_rank() over(order by HC_Ratio desc) as HC_Rank
    , relationship
    , MC_Count
    , MC_Ratio
    , dense_rank() over(partition by education order by MC_Ratio desc) as MC_Rank
    , race
    , LC_Count
    , LC_Ratio
    , dense_rank() over(partition by education, relationship order by LC_Ratio desc) as LC_Rank
from (
    select 
          education
        , sum(count(1)) over(partition by education) as HC_Count
        , sum(count(1)) over(partition by education) / sum(count(1)) over() as HC_Ratio
        , relationship
        , sum(count(1)) over(partition by education, relationship) as MC_Count
        , sum(count(1)) over(partition by education, relationship) / sum(count(1)) over(partition by education) as MC_Ratio
        , race
        , sum(count(1)) over(partition by education, relationship, race) as LC_Count
        , sum(count(1)) over(partition by education, relationship, race) / sum(count(1)) over(partition by education, relationship) as LC_Ratio
    from adult
    group by education, relationship, race
) A01
order by education, relationship, race
```
![image](https://user-images.githubusercontent.com/56889151/155879741-3d9a8903-6143-4b1b-9141-208d4cf9ab26.png)


### Pivot
`Univariate`
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

`multivariate`
```sql
select 
        relationship, education
      , sum(case when trim(race) = 'White' then 1 else 0 end) as White
      , sum(case when trim(race) = 'Black' then 1 else 0 end) as Black
      , sum(case when trim(race) = 'Asian-Pac-Islander' then 1 else 0 end) as API
      , sum(case when trim(race) = 'Amer-Indian-Eskimo' then 1 else 0 end) as AIE   
      , sum(case when trim(race) = 'Other' then 1 else 0 end) as Other      
from adult
group by relationship, education
order by 1, 2
```
![image](https://user-images.githubusercontent.com/56889151/155881698-37c57aec-7bf0-47d0-bc4f-9402b4d08f46.png)

