
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
`Categorical Univariate Frequency Analysis` : Order by high frequency
```sql
select 
      "education"                                                       as COL
    , sum(count(1)) over()                                              as NUM_ROWS      
    , education                                                         as INSTANCE
    , count(1)                                                          as CNT
    , sum(count(1)) over(order by count(1) desc)                        as CUMULATIVE_CNT    
    , sum(count(1)) over(partition by education) / sum(count(1)) over() as RATIO
    , sum(count(1)) over(order by count(1) desc) / sum(count(1)) over() as PERCENTILE    
    , dense_rank() over(order by count(1) desc)                         as D_RANK
    , count(1) over()                                                   as ROW_SHAPE    
from adult
group by education
order by CNT desc
```
![image](https://user-images.githubusercontent.com/56889151/155889524-c5b88ec5-1301-453d-83cd-abfe004cd985.png)

`Categorical Multivariate Frequency Analysis`
```sql
select 
      "education & relationship"                                                     as COL
    , education                                                                      as L1_INSTANCE
    , relationship                                                                   as L2_INSTANCE
    , sum(count(1)) over()                                                           as NUM_ROWS          
    , count(1)                                                                       as CNT
    , sum(count(1)) over(order by count(1) desc)                                     as CUMULATIVE_CNT
    , count(1) / sum(count(1)) over()                                                as RATIO
    , sum(count(1)) over(order by count(1) desc) / sum(count(1)) over()              as PERCENTILE    
    , count(1) over()                                                                as ROW_SHAPE        
from adult
group by education, relationship
order by CNT desc
```
![image](https://user-images.githubusercontent.com/56889151/155893151-eee40e1b-7411-4609-86ee-fdd6b482e609.png)

`Hierarchical Frequency Analysis`
```sql
select 
      education                                                                                                                   as L1_INSTANCE
    , sum(count(1)) over(partition by education)                                                                                  as L1_CNT_BY_GROUP
    , sum(count(1)) over(partition by education) / sum(count(1)) over()                                                           as L1_RATIO_BY_GROUP
    , relationship                                                                                                                as L2_INSTANCE
    , sum(count(1)) over(partition by education, relationship)                                                                    as L2_CNT_BY_GROUP
    , sum(count(1)) over(partition by education, relationship) / sum(count(1)) over(partition by education)                       as L2_RATIO_BY_GROUP
    , race                                                                                                                        as L3_INSTANCE
    , sum(count(1)) over(partition by education, relationship, race)                                                              as L3_CNT_BY_GROUP
    , sum(count(1)) over(partition by education, relationship, race) / sum(count(1)) over(partition by education, relationship)   as L3_RATIO_BY_GROUP
    , count(1) over()                                                                                                             as ROW_SHAPE            
from adult
group by education, relationship, race
order by education, relationship, race
```
![image](https://user-images.githubusercontent.com/56889151/155890600-078ec3db-fccd-43a8-b4cf-e3fbe10ef6d0.png)

```sql
select 
      L1_INSTANCE
    , L1_CNT_BY_GROUP
    , L1_RATIO_BY_GROUP
    , dense_rank() over(order by L1_RATIO_BY_GROUP desc) as L1_RANK_BY_GROUP
    , L2_INSTANCE
    , L2_CNT_BY_GROUP
    , L2_RATIO_BY_GROUP
    , dense_rank() over(order by L2_RATIO_BY_GROUP desc) as L2_RANK_BY_GROUP
    , L3_INSTANCE
    , L3_CNT_BY_GROUP
    , L3_RATIO_BY_GROUP
    , dense_rank() over(order by L3_RATIO_BY_GROUP desc) as L3_RANK_BY_GROUP
    , count(1) over()                                    as ROW_SHAPE            
from (
    select 
          education                                                                                                                   as L1_INSTANCE
        , sum(count(1)) over(partition by education)                                                                                  as L1_CNT_BY_GROUP
        , sum(count(1)) over(partition by education) / sum(count(1)) over()                                                           as L1_RATIO_BY_GROUP
        , relationship                                                                                                                as L2_INSTANCE
        , sum(count(1)) over(partition by education, relationship)                                                                    as L2_CNT_BY_GROUP
        , sum(count(1)) over(partition by education, relationship) / sum(count(1)) over(partition by education)                       as L2_RATIO_BY_GROUP
        , race                                                                                                                        as L3_INSTANCE
        , sum(count(1)) over(partition by education, relationship, race)                                                              as L3_CNT_BY_GROUP
        , sum(count(1)) over(partition by education, relationship, race) / sum(count(1)) over(partition by education, relationship)   as L3_RATIO_BY_GROUP
    from adult
    group by education, relationship, race
) A01
order by L1_INSTANCE, L2_INSTANCE, L3_INSTANCE
```
![image](https://user-images.githubusercontent.com/56889151/155890803-a3cacf8d-9a7b-4752-9731-718a5f87f95b.png)

### Conditional Frequency Analysis
```sql
select 
      "education"                                                       as COL
    , AGE_GROUP                                                         as CONDITIONAL_INSTANCE  
    , sum(count(1)) over()                                              as NUM_ROWS      
    , education                                                         as INSTANCE
    , count(1)                                                          as CNT
    , sum(count(1)) over(order by count(1) desc)                        as CUMULATIVE_CNT    
    , sum(count(1)) over(partition by education) / sum(count(1)) over() as RATIO
    , sum(count(1)) over(order by count(1) desc) / sum(count(1)) over() as PERCENTILE    
    , dense_rank() over(order by count(1) desc)                         as D_RANK
    , count(1) over()                                                   as ROW_SHAPE    
from (
    select 
          education
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
) A01
group by AGE_GROUP, education
order by CONDITIONAL_INSTANCE, CNT desc
```
![image](https://user-images.githubusercontent.com/56889151/155891216-b254ac5e-f583-4894-b26e-5c76fc12d561.png)



### Percentile Analysis
`Numerical Univariate Percentile Analysis`
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
![image](https://user-images.githubusercontent.com/56889151/155889576-17ac5d0d-2c54-4c8e-90d6-c2fd5f8a396b.png)

```sql
select 
      "age"                                                   as COL
    , sum(count(1)) over()                                    as NUM_ROWS    
    , age                                                     as INSTANCE
    , count(1)                                                as CNT
    , sum(count(1)) over(order by age)                        as CUMULATIVE_CNT
    , count(1) / sum(count(1)) over()                         as RATIO    
    , sum(count(1)) over(order by age) / sum(count(1)) over() as PERCENTILE
    , count(1) over()                                         as ROW_SHAPE
from adult
group by age
```
![image](https://user-images.githubusercontent.com/56889151/155889642-e4217137-43cd-402f-bc17-aa9cfbc5e9df.png)

`Binning`
```sql
select 
      "age"                                                       as COL
    , sum(sum(CNT)) over()                                        as ROW_NUMS
    , AGE_GROUP                                                   as INSTANCE_GROUP
    , count(1)                                                    as NUM_UNIQUE_INSTANCE
    , sum(CNT)                                                    as CNT
    , sum(sum(CNT)) over(order by AGE_GROUP)                      as CUMULATIVE_CNT
    , sum(CNT) / sum(sum(CNT)) over()                             as RATIO
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
![image](https://user-images.githubusercontent.com/56889151/155889717-afb65805-33b9-4dc6-b16b-ca8f4a720cae.png)


### Conditional Percentile Analysis
`Conditional Numerical Univariate Percentile Analysis`
```sql
select 
    "age"                          as COL
    , relationship                 as CONDITIONAL_INSTANCE
    , count(age)                   as NUM_ROWS
    , min(age)                     as MIN_VALUES
    , max(age)                     as MAX_VALUES    
    , sum(age)                     as SUM_VALUES
    , avg(age)                     as AVG_VALUES
    , sum(age) - 1.96*avg(age)     as LEFT_CONFIDENCE_INTERVAL
    , sum(age) + 1.96*avg(age)     as RIGHT_CONFIDENCE_INTERVAL
from adult
group by relationship
```
![image](https://user-images.githubusercontent.com/56889151/155890284-fe3e8111-23ea-4824-bc07-a1a04e32fb4a.png)

```sql
select 
      "age"                                                   as COL
    , relationship                                            as CONDITIONAL_INSTANCE
    , sum(count(1)) over()                                    as NUM_ROWS    
    , age                                                     as INSTANCE
    , count(1)                                                as CNT
    , sum(count(1)) over(order by age)                        as CUMULATIVE_CNT
    , count(1) / sum(count(1)) over()                         as RATIO    
    , sum(count(1)) over(order by age) / sum(count(1)) over() as PERCENTILE
    , count(1) over()                                         as ROW_SHAPE
from adult
group by relationship, age
```
![image](https://user-images.githubusercontent.com/56889151/155890268-c8af3c25-d751-4e47-9b4f-bbe3e06ad182.png)

`Conditional Binning`
```sql
select 
      "age"                                                       as COL
    , relationship                                                as CONDITIONAL_INSTANCE      
    , sum(sum(CNT)) over()                                        as ROW_NUMS
    , AGE_GROUP                                                   as INSTANCE_GROUP
    , count(1)                                                    as NUM_UNIQUE_INSTANCE
    , sum(CNT)                                                    as CNT
    , sum(sum(CNT)) over(order by AGE_GROUP)                      as CUMULATIVE_CNT
    , sum(CNT) / sum(sum(CNT)) over()                             as RATIO
    , sum(sum(CNT)) over(order by AGE_GROUP)/sum(sum(CNT)) over() as PERCENTILE    
    , dense_rank() over(order by sum(CNT) desc)                   as D_RANK
    , count(1) over()                                             as ROW_SHAPE    
from (
    select 
        age, relationship
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
    group by relationship, age
) A01
group by relationship, AGE_GROUP
order by relationship, AGE_GROUP
```
![image](https://user-images.githubusercontent.com/56889151/155890217-406c42be-fb00-45f7-982f-651f7b564973.png)
![image](https://user-images.githubusercontent.com/56889151/155890233-797e2656-143e-400c-8bc9-19f4ab136fc8.png)



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

