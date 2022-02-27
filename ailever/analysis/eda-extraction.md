
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


```sql
select 
      count(1)                                                          as NUM_ROWS
    , count(distinct age)                                               as NU_AGE 
    , count(distinct workclass)                                         as NU_WORKCLASS     
    , count(distinct fnlwgt)                                            as NU_FNLWGT 
    , count(distinct education)                                         as NU_EDUCATION 
    , count(distinct `education-num`)                                   as NU_EDUCATION_NUM 
    , count(distinct `marital-status`)                                  as NU_MARITAL_STATUS
    , count(distinct occupation)                                        as NU_OCCUPATION
    , count(distinct relationship)                                      as NU_RELATIONSHIP    
    , count(distinct race)                                              as NU_RACE        
    , count(distinct sex)                                               as NU_SEX   
    , count(distinct `capital-gain`)                                    as NU_CAPITAL_GAIN
    , count(distinct `capital-loss`)                                    as NU_CAPITAL_LOSS    
    , count(distinct `hours-per-week`)                                  as NU_HOURS_PER_WEEK
    , count(distinct `native-country`)                                  as NU_NATIVE_COUNTRY    
    , count(distinct 50K)                                               as NU_50K    
    , count(1) over()                                                   as ROW_SHAPE
from adult
```
![image](https://user-images.githubusercontent.com/56889151/155889008-6d796020-d41f-4bbd-bc24-6fe02f938c66.png)


### Frequency Analysis
`Categorical Univariate Frequency Analysis`
```sql
select 
      "education"                                                       as COL
    , education                                                         as INSTANCE
    , count(1)                                                          as CNT
    , sum(count(1)) over(partition by education) / sum(count(1)) over() as PERCENTILE
    , dense_rank() over(order by count(1) desc)                         as D_RANK
    , count(1) over()                                                   as ROW_SHAPE    
from adult
group by education
```
![image](https://user-images.githubusercontent.com/56889151/155886861-362c5f4f-8396-418d-8ac8-c6baacf2e386.png)

`Categorical Multivariate Frequency Analysis`
```sql
select 
      education                                                                                             as L1_INSTANCE
    , relationship                                                                                          as L2_INSTANCE
    , sum(count(1)) over(partition by education)                                                            as L1_CNT_BY_GROUP    
    , count(1)                                                                                              as L2_CNT_BY_GROUP
    , sum(count(1)) over(partition by education) / sum(count(1)) over()                                     as L1_RATIO_BY_GROUP  
    , sum(count(1)) over(partition by education, relationship) / sum(count(1)) over(partition by education) as L2_RATIO_BY_GROUP  
    , dense_rank() over(partition by education order by count(1))                                           as L2_D_RANK_BY_GROUP 
    , count(1) over()                                                                                       as ROW_SHAPE        
from adult
group by education, relationship
```
![image](https://user-images.githubusercontent.com/56889151/155887170-12b5f421-2716-4909-a265-9da7abe173eb.png)



`Hierarchical Frequency Analysis`
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


### Percentile Analysis
`Numerical Univariate Percentile`
```sql
select 
      "age"                        as COL
    , count(age)                   as NUM_ROWS
    , min(age)                     as MIN_VALUES
    , max(age)                     as MAX_VALUES    
    , sum(age)                     as SUM_VALUES
    , avg(age)                     as AVG_VALUES
    , sum(age) - 1.96*avg(age)     as LEFT_CONFIDENCE_INTERVAL
    , sum(age) + 1.96*avg(age)     as RIGHT_CONFIDENCE_INTERVAL
from adult
```
![image](https://user-images.githubusercontent.com/56889151/155886707-af0e90f9-15e3-466e-9a0b-6aa1182c416a.png)

```sql
select 
      "age"                                                   as COL
    , sum(count(1)) over()                                    as NUM_ROWS    
    , age                                                     as INSTANCE
    , count(1)                                                as CNT
    , sum(count(1)) over(order by age)                        as CUMULATIVE_CNT
    , sum(count(1)) over(order by age) / sum(count(1)) over() as PERCENTILE
    , count(1) over()                                         as ROW_SHAPE
from adult
group by age
```
![image](https://user-images.githubusercontent.com/56889151/155886554-8e59ff31-4bdc-4e52-9204-71a67c31cd56.png)

`Binning`
```sql
select 
      "age"                                                       as COL
    , sum(sum(CNT)) over()                                        as ROW_NUMS
    , AGE_GROUP                                                   as INSTANCE_GROUP
    , count(1)                                                    as NUM_UNIQUE_INSTANCE
    , sum(CNT)                                                    as CNT
    , sum(sum(CNT)) over(order by AGE_GROUP)                      as CUMULATIVE_CNT
    , sum(sum(CNT)) over(order by AGE_GROUP)/sum(sum(CNT)) over() as PERCENTILE    
    , dense_rank() over(order by sum(CNT) desc)                   as D_RANK
    , count(1) over()                                             as ROW_SHAPE    
from (
    select 
        age
        , count(1) as CNT
        , case when age >= 10 and age < 20 then 10
               when age >= 20 and age < 30 then 20
               when age >= 30 and age < 40 then 30
               when age >= 40 and age < 50 then 40
               when age >= 50 and age < 60 then 50
               when age >= 60 and age < 70 then 60
               when age >= 70 and age < 80 then 70
               when age >= 80 and age < 90 then 80
               else 90 end as AGE_GROUP
    from adult
    group by age
) A01
group by AGE_GROUP
order by AGE_GROUP
```
![image](https://user-images.githubusercontent.com/56889151/155887907-e7adf5ca-d4a9-4212-a73d-7aa25db75209.png)


`Conditional Percentile Analysis`
```sql
select 
      relationship
    , count(age)
    , min(age)
    , max(age)    
    , sum(age)
    , avg(age)
    , sum(age) - 1.96*avg(age)
    , sum(age) + 1.96*avg(age) 
from adult
group by relationship
```
![image](https://user-images.githubusercontent.com/56889151/155882947-ef4fa202-ab55-4ba3-aabf-42d169cec278.png)



### Pivot
`Categorical Univariate`
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

`Numerical Univariate`
```sql
select 
      education
      , dense_rank() over(order by count(1) desc)   as D_RANK
      , count(1)                                    as CNT
      , percent_rank() over(order by count(1) desc) as P_RANK
      , min(age)
      , max(age)
      , sum(case when age >= 10 and age < 20 then 1 else 0 end) as teenager
      , sum(case when age >= 20 and age < 30 then 1 else 0 end) as twenties
      , sum(case when age >= 30 and age < 40 then 1 else 0 end) as thirties
      , sum(case when age >= 40 and age < 50 then 1 else 0 end) as forties     
      , sum(case when age >= 50 and age < 60 then 1 else 0 end) as fifties  
      , sum(case when age >= 60 and age < 60 then 1 else 0 end) as sixties
      , sum(case when age >= 70 and age < 80 then 1 else 0 end) as seventies  
      , sum(case when age >= 80 and age < 90 then 1 else 0 end) as eighties   
      , sum(case when age >= 90 then 1 else 0 end) as etc      
      , count(1) as allages
from adult
group by education
```
![image](https://user-images.githubusercontent.com/56889151/155884459-11633f3f-caba-40ae-b706-88ddefddf6f3.png)


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

