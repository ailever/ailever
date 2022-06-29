- https://dev.mysql.com/doc/
- https://dev.mysql.com/doc/refman/8.0/en/language-structure.html

---

## Listener Check

`LINUX`
```bash
$ ps aux
$ netstat -nlpt
$ sudo service --status-all
# sudo service mysql [start/stop/restart]
```

`WINDOWS`
```bash
netstat -ano | findstr TCP
lsnrctl services
lsnrctl status
```

---

## Installation
```bash
$ sudo apt install mysql-server
$ sudo mysql -v
```

---

## MySQL for python
```bash
$ pip install pymysql
```
```python
import pandas as pd
import pymysql

connection = pymysql.connect(host='localhost', port=[port], user='[user_id]', password='[password]', db='[database]', charset='utf8')
cursor = connection.cursor()

query = 'select * from adult'
cursor.execute(query)
rows = cursor.fetchall()
connection.close()

data = pd.DataFrame(rows)
```
```python
import pandas as pd
import pymysql

connection = pymysql.connect(host='localhost', user='[user_id]', password='[password]', db='[database]', charset='utf8')

query = 'select * from adult'
data = pd.read_sql_query(query, connection)
data
```

---


## Account Management
### Login to mysql server
```bash
$ sudo mysql -u root
$ sudo mysql -u root -p
```
```sql
use mysql;
```

### User inquery
```sql
SELECT user, host, authentication_string, plugin FROM user;
```

### Add user
`create user`
```sql
CREATE user '[account]'@'[ip]' identified by '[passwd]';
```
```sql
CREATE user 'test'@'localhost' identified by 'passwd';
CREATE user 'test'@'%' identified by 'passwd';
```

`grant all privileges on`
```sql
GRANT all privileges ON [dbname].* to [account]@'[ip]' identified by '[passwd]';
```
```sql
CREATE all privileges ON samdb01.* to test_account@'localhost' identified by 'passwd';
CREATE all privileges ON samdb01.* to test_account@'%' identified by 'passwd';
```



### Privileges
```sql
GRANT all privileges ON [dbname].* to [account]@'[ip]';
```
```sql
GRANT all privileges ON samdb01.* to test_account@'localhost';
GRANT all privileges ON samdb01.* to test_account@'%';
```


### Change password
```sql
ALTER USER '[account]'@'[ip]' IDENTIFIED WITH mysql_native_password BY '[passwd]';
flush privileges;
```
```sql
ALTER USER 'test_account'@'localhost' IDENTIFIED WITH mysql_native_password BY '[passwd]';
flush privileges;
```

### Delete account
```sql
DELETE FROM user WHERE user='[account]';
flush privileges;
```
```sql
DELETE FROM user WHERE user='test_account';
flush privileges;
```

---

# DATABASE
## Meta Table
```sql
SHOW databases;
SHOW tables;
```

```sql
SELECT user();
SELECT current_user();
```

```bash
use mysql;
select host, user, authentication_string from user;
alter user '[id]'@'[ip]' identified with mysql_native_password by '[password]';
select user();
select current_user();
show databases;
use [database];
show tables;
select database();
```

## Encoding
`CREATE` : database
```bash
create database [db_name] default character set utf8 collate utf8_general_ci;
```

`CREATE` : table
```bash
mysql> create table [table_name] (
    -> id int(4) primary key,
    -> ...
    -> ) default character set utf8 collate utf8_general_ci;
```

`ALTER` : database
```bash
alter database [db_name] default character set utf8 collate utf8_general_ci;
```

`ALTER` : table
```bash
alter table [table_name] default character set utf8 collate utf8_general_ci;
```


---

# TABLE
## Syntax
Task Order: FROM > ON > JOIN > WHERE > GROUP BY > HAVING > SELECT > DISTINCT > ORDER BY > OFFSET > LIMIT 
Syntax Order: SELECT > FROM > JOIN > ON > WHERE > GROUP BY > HAVING > ORDER BY > OFFSET > LIMIT


### CREATE
`table`
```sql
CREATE TABLE [table] (
    [column1] INT,
    [column2] VARCHAR(255),
    [column3] VARCHAR(255),
    [column4] VARCHAR(255),
    [column5] VARCHAR(255)
);
```

`index`
```sql
CREATE INDEX [index_name] ON [table] ([column1], [column2], ...);
SHOW INDEX FROM [table];
DESC [table];
```

### INSERT INTO
```sql
INSERT INTO [table]
VALUES (1, 'col2_instance', 'col3_instance', 'col4_instance', 'col5_instance');
```
```sql
INSERT INTO [table] ([column2], [column3])
VALUES ('col2_instance', 'col3_instance');
```

### DROP
```sql
DROP TABLE [table];
```

### DESC
```sql
DESC [table];
```

### SELECT FROM
```sql
SET @idx:=0;
SELECT @idx:=@idx+1 AS idx, [table].* FROM [table];
```
`group_concat`
```sql
SELECT [column], GROUP_CONCAT([column]) FROM [table] GROUP BY [column];
```
`count & distinct`
```sql
SELECT [base_column], COUNT(DISTINCT [another_column]) FROM [table] GROUP BY [base_column];
```

`rank & dense_rank & row_number & ntile`
```sql
SELECT 
      rank()         over(order by [column])
    , dense_rank()   over(order by [column])
    , row_number()   over(order by [column])
    , percent_rank() over(order by [column])    
    , ntile(100)     over(order by [column])
FROM [table]
```


#### GROUP BY
```sql
SELECT * FROM adult
```
![image](https://user-images.githubusercontent.com/56889151/155865692-981285c1-553c-46eb-9ea4-7fcd6204c6de.png)
##### COUNT OVER()
`(ROW_NUMBER) Ordinal Number` vs `(COUNT) Cardinal Number`
```sql
SELECT 
      sex
    , race
    ,     count(1) over(order by sex)
    ,     count(1) over(partition by sex)
    ,     count(1) over(partition by sex order by race)    
    , row_number() over(partition by sex order by race)
    ,     count(1) over(order by race)
    ,     count(1) over(partition by race)
    ,     count(1) over(partition by race order by sex)
    , row_number() over(partition by race order by sex)
    ,     count(1)
    ,     count(1) over()
    ,     count(1) over(order by sex, race)
    , row_number() over(order by sex, race)    
FROM adult
GROUP BY 1, 2
ORDER BY 1, 2
```
![image](https://user-images.githubusercontent.com/56889151/155869978-6043edf2-4f4f-4e24-a011-0d8e862cade0.png)

```sql
SELECT 
      sex
    , race
    , 50K
    ,     count(1) over(order by sex)
    ,     count(1) over(partition by sex)
    ,     count(1) over(partition by sex order by race)    
    , row_number() over(partition by sex order by race)
    ,     count(1) over(order by race)
    ,     count(1) over(partition by race)
    ,     count(1) over(partition by race order by sex)
    , row_number() over(partition by race order by sex)
    ,     count(1)
    ,     count(1) over()
    ,     count(1) over(order by sex, race)
    , row_number() over(order by sex, race)    
FROM adult
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
```
![image](https://user-images.githubusercontent.com/56889151/155870074-b4fee3ac-ea4e-4945-bf87-3fd8b5fdf102.png)

##### SUM OVER()
```sql
SELECT 
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
FROM adult
GROUP BY 1, 2
ORDER BY 1, 2
```
![image](https://user-images.githubusercontent.com/56889151/155869554-1cc82776-9c86-4e9f-8397-7d101103770e.png)


```sql
SELECT 
      sex
    , race
    , count(1) over(order by sex)
    , sum(0) over(order by sex) 
    , sum(1) over(order by sex)
    , sum(2) over(order by sex)
    , count(1) over(order by race)
    , sum(0) over(order by race) 
    , sum(1) over(order by race)
    , sum(2) over(order by race)
    , count(1) over(order by sex, race)        
    , sum(0) over(order by sex, race)    
    , sum(1) over(order by sex, race)    
    , sum(2) over(order by sex, race)    
FROM adult
GROUP BY 1, 2
ORDER BY 1, 2
```
![image](https://user-images.githubusercontent.com/56889151/155869814-8dede706-fc1a-4e8c-b7e3-6a3e26ab5524.png)

```sql
SELECT 
      sex
    , race
    , count(1)
    , sum(count(1)) over()
    , sum(count(1)) over(order by sex)    
    , sum(count(1)) over(order by sex, race)
    , sum(count(1)) over(partition by sex)
    , sum(count(1)) over(partition by sex order by race)
    , sum(count(1)) over(partition by race)
    , sum(count(1)) over(partition by race order by sex)
    , sum(count(1)) over(partition by sex, race)
    , sum(count(1)) over(partition by sex, race order by sex)
    , sum(count(1)) over(partition by sex, race order by race)
    , sum(count(1)) over(partition by sex, race order by sex, race)    
FROM adult
GROUP BY 1, 2
ORDER BY 1, 2
```
![image](https://user-images.githubusercontent.com/56889151/155873889-0dcd1494-f4c2-4d2b-9416-8cea14b3d078.png)

##### RANK OVER(), DENSE_RANK OVER(), ROW_NUMBER() OVER(), NTILE(n) OVER() 
```sql
SELECT 
      sex
    , race
    , count(1)
    , rank()       over(order by count(1))
    , dense_rank() over(order by count(1))
    , row_number() over(order by count(1))
    , ntile(10)    over(order by count(1))
FROM adult
GROUP BY 1, 2
ORDER BY 1, 2
```
![image](https://user-images.githubusercontent.com/56889151/155879362-06ead136-b780-4392-b6da-cbadcb47692e.png)




### DELETE

### ALTER
`drop`
```sql
ALTER TABLE [table]
DROP COLUMN [column];
```

---

## Data Analysis
### Preprocessing
`missing values`
```sql
```

`duplicated values`
```sql
SELECT [column1], [column2], [column3], ..., COUNT(1)
FROM [table] 
GROUP BY [column1], [column2], [column3], ...
ORDER BY [column1], [column2], [column3], ...;
```

`numbering`
```sql
SELECT @idx := 0;
SELECT @idx := @idx+1 AS ROWNUM, [column1], [column2], [column3], ... FROM [table];
```


### Frequency Analysis
```sql
SELECT 
    [column], 
    COUNT([column])
FROM
    [table]
GROUP BY [column];
```

```sql
WITH LowLevelClassification AS (SELECT [h_column], [m_column], [l_column], count(1) AS LL1_CNT FROM [table] GROUP BY [h_column], [m_column], [l_column] ORDER BY [h_column], [m_column], [l_column])
, MiddleLevelClassification AS (SELECT [h_column], [m_column], count(1)             AS LL2_CNT FROM [table] GROUP BY [h_column], [m_column]             ORDER BY [h_column], [m_column])
  , HighLevelClassification AS (SELECT [h_column], count(1)                         AS LL3_CNT FROM [table] GROUP BY [h_column]                         ORDER BY [h_column])

SELECT 
       LowLevelClassification.[h_column], 
      HighLevelClassification.LL3_CNT, 
       LowLevelClassification.[m_column], 
    MiddleLevelClassification.LL2_CNT, 
       LowLevelClassification.[l_column], 
       LowLevelClassification.LL1_CNT 
FROM LowLevelClassification 
LEFT JOIN MiddleLevelClassification ON LowLevelClassification.[m_column] = MiddleLevelClassification.[m_column] 
LEFT JOIN   HighLevelClassification ON LowLevelClassification.[h_column] =   HighLevelClassification.[h_column] 
ORDER BY 
    LowLevelClassification.[h_column], 
    LowLevelClassification.[m_column], 
    LowLevelClassification.[l_column];
```


### Percentile Analysis
```sql
```

---

## Reference
