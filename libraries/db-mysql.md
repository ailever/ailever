- https://dev.mysql.com/doc/

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


## Account Management
### Login to mysql server
```bash
$ sudo mysql -u root
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

## Metatable
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


---

## Syntax
Task Order: FORM > ON > JOIN > WHERE > GROUP BY > HAVING > SELECT > ORDER BY   
Syntax Order: SELECT > FROM > JOIN > ON > WHERE > GROUP BY > HAVING > ORDER BY


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
set @idx:=0;
select @idx:=@idx+1 as idx, [table].* from [table];
```

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
