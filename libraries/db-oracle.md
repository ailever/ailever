## Oracle Database
![image](https://user-images.githubusercontent.com/56889151/146068105-57be9534-5226-4f61-be16-228265016a08.png)
![image](https://user-images.githubusercontent.com/56889151/146070120-718d7d77-2646-4d79-9fbf-217336af12cf.png)

[Oracle Database Docs](https://docs.oracle.com/en/database/oracle/oracle-database/index.html)
- [Oracle Database Concepts](https://docs.oracle.com/cd/E11882_01/server.112/e40540/preface.htm#CNCPT88774)
- [SQL Language Quick Reference](https://docs.oracle.com/en/database/oracle/oracle-database/21/sqlqr/index.html)
- [Oracle Database Performance Tuning Guide](https://docs.oracle.com/database/121/TGDBA/preface.htm#TGDBA463)



## Oracle Network
- `WINDOWS`: C:\app\user\product\21c\homes\OraDB21Home1\network\admin
	- listener.ora  >  HOST, IP
	- tnsnames.ora  >  SERVICE_NAME: SID  

### Listener Check  

`LINUX`
```bash
$ netstat -nlpt
$ sudo service --status-all
```
`WINDOWS`
```powershell
netstat -ano | findstr TCP
lsnrctl services
lsnrctl status
```

---



## cx_Oracle for python
- [cx_Oracle](https://oracle.github.io/python-cx_Oracle/)
- [Oracle Instant-Client](https://www.oracle.com/database/technologies/instant-client.html) [LINUX/WINDOWS]

`UBUNTU`
```bash
$ wget [oracle instance client file path]
$ sudo apt install alien
$ sudo alien -i [file.rpm]
```


- https://www.oracle.com/database/technologies/appdev/python/quickstartpythononprem.html

```bash
$ python -m pip install cx_Oracle --upgrade --user
```
```python
import cx_Oracle
import os, sys

# Set your information about Oracle
oracle_instantclient_path = ~
user = ~
password = ~
dsn = f'{IP~}:{PORT~}/{SID~}'

sys.path.append(oracle_instantclient_path)
os.putenv('NLS_LANG', 'AMERICAN_AMERICA.UTF8')
connection = cx_Oracle.connect(
    user=user,
    password=password,
    dsn=dsn)

cursor = connection.cursor()
cursor.execute("""select * from user_tables""")
for row in cursor:
    print(row)
    
cursor.close()
connection.close()
```

---

## Metatable
### Table Information
```sql
SELECT * FROM all_tables WHERE owner = '[username]';
SELECT * FROM user_tables;
SELECT * FROM tabs;
SELECT * FROM cols;
```

### Pesudo Column
```sql
SELECT ROWNUM FROM [table];
```

### Dummy Column
```sql
SELECT [operation] FROM DUAL;
```

### User Indexes
```sql
SELECT * FROM user_indexes;
```

---

## Syntax
Task Order: FORM > ON > JOIN > WHERE > GROUP BY > HAVING > SELECT > ORDER BY 
Syntax Order: SELECT > FROM > JOIN > ON > WHERE > GROUP BY > HAVING > ORDER BY


### SELECT
#### SELECT FROM
```sql
SELECT * FROM [table];
SELECT [column] FROM [table];
SELECT [instance] FROM [table];
```

#### SELECT FROM SAMPLE
```sql
SELECT [column] FROM [table] sample(20);   -- sampling 20% from population
```

#### SELECT CASE WHEN THEN END AS FROM
```sql
SELECT [column],
       CASE WHEN [column] > '10' AND [column] <= '50' THEN 'A'
	    WHEN [column] > '50' AND [column] <= '90' THEN 'B' 
	    ELSE 'C' END AS label
FROM [table];
```

#### SELECT FROM WHERE LIKE
```sql
```

#### SELECT FROM GROUP BY HAVING
```sql
SELECT [column1], count([column2]) FROM GROUP BY [column1] HAVING [condition for aggregation column1];
SELECT [base column], [aggregation]([target column]) FROM GROUP BY [base column] HAVING [condition for aggregration base column];
```

#### WITH SELECT FROM
```sql
WITH [new_table] AS (SELECT * FROM [origin_table])
SELECT * FROM [new_table];
```
```sql
WITH [new_table1]  AS (SELECT * FROM [origin_table1]),
     [new_table2]  AS (SELECT * FROM [origin_table2]),
     [new_table3]  AS (SELECT * FROM [new_table1]),
     [new_table3]  AS (SELECT * FROM [new_table2]),
     [final_table] AS (SELECT * FROM [new_table3], [new_table4]),     
SELECT * FROM [final_table] ;
```


### CREATE
- [Data types](https://docs.oracle.com/cd/A58617_01/server.804/a58241/ch5.htm)

`create`
```sql
CREATE TABLE [table] (
	[column1] VARCHAR(255),
	[column2] VARCHAR(255),
	[column3] INTEGER,
	[column4] NUMBER(10,2),
	[column5] DATE
);
```

`copy`
```sql
CREATE TABLE [new_table] AS (SELECT * FROM [origin_table]);                   -- copy table
CREATE TABLE [new_table] AS (SELECT * FROM [origin_table] WHERE 1=0);         -- copy table to have empty value
```

#### Constraints
```sql
CREATE TABLE constraint_table (
    column1 VARCHAR2(10),
    column2 VARCHAR2(10) NOT NULL,
    column3 VARCHAR2(10) UNIQUE,
    column4 VARCHAR2(10) UNIQUE NOT NULL,     
    CONSTRAINTS column5 UNIQUE (column2) 
);
```

### UPDATE
`add/delete column`
```sql
ALTER TABLE [table] ADD [new_column] [data-type];                                     -- add column
ALTER TABLE [table] ADD [new_column] [data-type] DEFAULT [default-value] NOT NULL;    -- add column with options
ALTER TABLE [table] DROP COLUMN [column];                                             -- delete column
```

`update`
```sql
UPDATE [table] SET [dummy_column] = 10*[origin_column];                                 -- set dummy column as operation about origin column
```

`change column's order`
```sql
ALTER TABLE [table] MODIFY [column] INVISIBLE;
ALTER TABLE [table] MODIFY [column] VISIBLE;
```

### INSERT
```sql
```

### DROP
```sql
DROP TABLE [table];
```







### User-defined function
```sql
```

### Iteration
```sql
```

### Condition Statement
```sql
```

--- 

## Table Control
### JOIN
![image](https://user-images.githubusercontent.com/56889151/145710327-e2fbdeb6-b922-4ab8-834c-96bb673a9065.png)
#### Cartesian Join
```sql
SELECT table1.*, table2.* FROM [table1], [table2];
```
#### EquiJoin
```sql
```
#### Non-EquiJoin
```sql
```
#### Self Join
```sql
```
#### Left Join
```sql
```
#### Right Join
```sql
```
#### Inner Join
```sql
```
#### Outer Join
```sql
```




### UNION
```sql
SELECT * FROM [table1]
UNION ALL
SELECT * FROM [table2];
```



---

## Functions
### Type Casting
```sql
SELECT 
    TO_NUMBER([column]), 
      TO_CHAR([column]), 
      TO_DATE([column], 'YYYY') 
FROM [table];
```
```sql
SELECT 
    TO_DATE('20210131'),             -- 2021-01-31 00:00:00.000 (Current Date: 2021-12-12)
    TO_DATE(20210131),               -- 2021-01-31 00:00:00.000 (Current Date: 2021-12-12)
    TO_DATE('2021', 'YYYY'),         -- 2021-12-01 00:00:00.000 (Current Date: 2021-12-12)
    TO_DATE('01', 'MM'),             -- 2021-01-01 00:00:00.000 (Current Date: 2021-12-12)
    TO_DATE('31', 'DD'),             -- 2021-12-31 00:00:00.000 (Current Date: 2021-12-12)
    TO_DATE('20210131', 'YYYYMMDD'), -- 2021-01-31 00:00:00.000 (Current Date: 2021-12-12)
    TO_DATE(20210101, 'YYYYMMDD')    -- 2021-01-01 00:00:00.000 (Current Date: 2021-12-12)
FROM DUAL;
```

`TO CATEGORICAL VARIABLE` or `TO LABEL`
```sql
SELECT CASE WHEN [numerical_column]=1 THEN 'A'
            WHEN [numerical_column]=2 THEN 'B'
            WHEN [numerical_column]=3 THEN 'C'
            ELSE 'etc' END 
FROM [table];
```

`TO NUMERICAL VARIABLE`
```sql
SELECT CASE WHEN [categorical_column]='A' THEN 1
            WHEN [categorical_column]='B' THEN 2
            WHEN [categorical_column]='C' THEN 3
            ELSE 0 END 
FROM [table];
```
```sql
SELECT 
    [categorical_column],
    count([categorical_column]),
    sum(CASE WHEN [categorical_column]='instance' THEN 1 ELSE 0 END)
FROM [table]
GROUP BY [categorical_column];
```
```sql
SELECT 
    [categorical_column1],
    count([categorical_column2]),
    sum(CASE WHEN [categorical_column3]='instance' THEN 1 ELSE 0 END)
FROM [table]
GROUP BY [categorical_column1];
```




### VARCHAR
- https://docs.oracle.com/cd/B19306_01/B14251_01/adfns_regexp.htm

```sql
SELECT LENGTH('ABC') FROM dual;
```

```sql
SELECT replace('ABC', 'A', 'B') FROM dual;  -- 'BBC'
```

```sql
SELECT instr('123456789', '7') FROM dual;  -- '7'
SELECT instr('123456789', '8') FROM dual;  -- '8'
SELECT instr('123456789', '9') FROM dual;  -- '9'

SELECT instr('1234567890123456789', '7', 1, 1) FROM dual;   -- '7' : Find FIRST 7  from  1st position in string
SELECT instr('1234567890123456789', '7', 10, 1) FROM dual;  -- '17': Find FIRST 7  from 10th position in string
SELECT instr('1234567890123456789', '7', 1, 2) FROM dual;   -- '17': Find SECOND 7 from  1st position in string
```

```sql
SELECT substr('123456789', 7)   FROM dual; -- '789'
SELECT substr('123456789', -3)  FROM dual; -- '789'
SELECT substr('123456789', 7,3) FROM dual; -- '789'
```

```sql
-- split
SELECT 
    SUBSTR('ABC_DEF', 1, INSTR('ABC_DEF', '_' , 1, 1)-1),
    SUBSTR('ABC_DEF', INSTR('ABC_DEF', '_' , 1, 1)+1)
FROM dual;  -- 'ABC', 'DEF'
```

```sql
SELECT 'ABC'||'DEF' FROM dual;
SELECT concat('ABC', 'DEF') FROM dual;
SELECT [column1] || '  SPACE  ' || [column2] FROM [table];
```

```sql
SELECT 
     TRIM('      SPACE      '),   -- 'SPACE'
    LTRIM('      SPACE      '),   -- 'SPACE      '
    RTRIM('      SPACE      ')    -- '      SPACE'
FROM dual;
```

```sql
SELECT CASE WHEN [categorical_column] LIKE '%A%' THEN 1
            WHEN [categorical_column] LIKE '%B%' THEN 2
            WHEN [categorical_column] LIKE '%C%' THEN 3
            ELSE 0 END 
FROM [table];
```

`REGEXP_LIKE`
```sql
```

`REGEXP_REPLACE`
```sql
```

`REGEXP_INSTR`
```sql
```

`REGEXP_SUBSTR`
```sql
```


### NUMBER
```sql
SELECT 
     count([column]) AS count,
       sum([column]) AS sum,
       avg([column]) AS avg,
    stddev([column]) AS stddev,	
    median([column]) AS median,
       min([column]) AS min,
       max([column]) AS max
FROM [table];
```

### DATE
```sql
```

---

## Analysis
- [Statistics](https://www.oracle.com/database/technologies/bi-datawarehousing.html)

### Integrity
#### Check Primary Key(Identifier)
```sql
```
#### Check Referential Key
```sql
```

### Duplication
```sql
  SELECT [column],
         COUNT(*) AS DUPLICATION_COUNT
    FROM [table]
GROUP BY [column]
HAVING COUNT(*) > 1 ;
```

### Null-Counting
```sql
WITH nullcounting AS (
	SELECT count([column]) rnefnv,
	       count(*)        rn 
	FROM [table])
SELECT rn - rnefnv 
FROM nullcounting;        -- rn: rownum, rnefnv: rownum_except_for_null_values
```

```sql
SELECT count(*) FROM [table] WHERE [column] IS NULL;
```


### Cleaning
```sql
SELECT * FROM [table] WHERE [column] IS NOT NULL;
```

### Unique Instance-Check
```sql
SELECT DISTINCT [column] FROM [table];
SELECT [column], count([column]) FROM [table] GROUP BY [column] ;
```

### Data Range
```sql
```

### Numbering
`ROWNUM without ORDER BY`
```sql
-- [SUMMARY]: ROWNUM
SELECT ROWNUM, [table].* FROM [table];
```
`ROWNUM with ORDER BY`
```sql
-- [SUMMARY]: WRAPPING
SELECT ROWNUM, numbering.* FROM ( 
	SELECT *
	FROM [table]
	ORDER BY [column]) numbering;
```
`ROW_NUMBER() OVER(~) with ORDER BY`
```sql
-- [SUMMARY]: ROW_NUMBER() OVER(ORDER BY [column] )
SELECT ROW_NUMBER() OVER(ORDER BY [column1]) row_num, [table].*
FROM [table]
ORDER BY [column1];

SELECT ROW_NUMBER() OVER(ORDER BY [column1], [column2]) row_num, [table].*
FROM [table]
ORDER BY [column1], [column2];

SELECT ROW_NUMBER() OVER(ORDER BY [column1], [column2], [column3]) row_num, [table].*
FROM [table]
ORDER BY [column1], [column2], [column3];
```

### Numbering by group partition
```sql
-- [SUMMARY]: ROW_NUMBER() OVER(PARTITION BY [criterion_column] ORDER BY [column1], [column2], [column3])
SELECT ROW_NUMBER() OVER(PARTITION BY [criterion_column] ORDER BY [column1], [column2], [column3]) row_num, [table].*
FROM [table]
ORDER BY [column1], [column2], [column3];
```

### Ranking
```sql
-- [SUMMARY]:       RANK() OVER(ORDER BY [column] DESC)
-- [SUMMARY]: DENSE_RANK() OVER(ORDER BY [column] DESC)
SELECT
    [column], 
          RANK() OVER (ORDER BY [column] DESC)       rank,
    DENSE_RANK() OVER (ORDER BY [column] DESC) dense_rank
FROM [table] 
ORDER BY [column] DESC;
```


### Ranking by group partition
```sql
-- [SUMMARY]:       RANK() OVER (PARTITION BY [criterion_column] ORDER BY [ranking_target_column] DESC)
-- [SUMMARY]: DENSE_RANK() OVER (PARTITION BY [criterion_column] ORDER BY [ranking_target_column] DESC)
SELECT 
    [criterion_column],
          RANK() OVER (PARTITION BY [criterion_column] ORDER BY [ranking_target_column] DESC)       rank, 
    DENSE_RANK() OVER (PARTITION BY [criterion_column] ORDER BY [ranking_target_column] DESC) dense_rank
FROM [table] 
ORDER BY [criterion_column], [ranking_target_column] DESC;
```

### Minmax by group partition
```sql
SELECT 
         [criterion_column], 
     MIN([target_column]) KEEP(DENSE_RANK FIRST ORDER BY [target_column]) OVER(PARTITION BY [criterion_column]) min,
     MAX([target_column]) KEEP(DENSE_RANK  LAST ORDER BY [target_column]) OVER(PARTITION BY [criterion_column]) max
FROM [table]
ORDER BY [criterion_column], [target_column] DESC;
```

### Conceptual clustering based on statistics
```sql
WITH attaching_table AS (
    SELECT [column], sum(CASE WHEN [column]='instance' THEN 1 ELSE 0 END)
    FROM [table] GROUP BY [column])
SELECT * FROM [table] LEFT JOIN attaching_table ON [table].[column]=attaching_table.[column];
```

### Binning
```sql
```

### Statistics
```sql
```

### Derivatives features
```sql
```


---

## Database Basic Concept
### Integrity
- **Entity integrity** concerns the concept of a primary key. Entity integrity is an integrity rule which states that every table must have a primary key and that the column or columns chosen to be the primary key should be unique and not null.
- **Referential integrity** concerns the concept of a foreign key. The referential integrity rule states that any foreign-key value can only be in one of two states. The usual state of affairs is that the foreign-key value refers to a primary key value of some table in the database. Occasionally, and this will depend on the rules of the data owner, a foreign-key value can be null. In this case, we are explicitly saying that either there is no relationship between the objects represented in the database or that this relationship is unknown.
- **Domain integrity** specifies that all columns in a relational database must be declared upon a defined domain. The primary unit of data in the relational data model is the data item. Such data items are said to be non-decomposable or atomic. A domain is a set of values of the same type. Domains are therefore pools of values from which actual values appearing in the columns of a table are drawn.
- **User-defined integrity** refers to a set of rules specified by a user, which do not belong to the entity, domain and referential integrity categories.

`Entity integrity`
```sql
CREATE TABLE entity_integrity (
    column1 VARCHAR2(10) PRIMARY KEY, 
    column2 VARCHAR2(10) 
); 
```

`Referential integrity`
```sql
CREATE TABLE entity_integrity ( 
    parent_pk number primary key 
); 
CREATE TABLE referential_integrity ( 
    child_pk number primary key, 
    parentid number, 
    foreign key (parentid) references entity_integrity(parent_pk)
);
```

`Domain integrity`/`User-defined integrity`
```sql
CREATE TABLE domain_integrity (
    column1 NUMBER       CONSTRAINTS check1 CHECK ( column1 BETWEEN 1 AND 9),
    column2 VARCHAR2(10) CONSTRAINTS check2 CHECK ( column2 IN ('MALE', 'FEMALE')) 
); 
```


### Transaction
- Automicity
- Consistency
- Isolation
- Durability


--- 

## Reference

- https://www.youtube.com/playlist?list=PLVsNizTWUw7FzFgU1qe-n7_M7eMFA9d-f

