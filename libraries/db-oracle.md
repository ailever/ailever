- `WINDOWS`: C:\app\user\product\21c\homes\OraDB21Home1\network\admin
	- listener.ora  >  HOST, IP
	- tnsnames.ora  >  SERVICE_NAME: SID
- [Data types](https://docs.oracle.com/cd/A58617_01/server.804/a58241/ch5.htm)
- [Statistics](https://www.oracle.com/database/technologies/bi-datawarehousing.html)

---



## cx_Oracle
- [cx_Oracle](https://oracle.github.io/python-cx_Oracle/)
- [Oracle Instant-Client](https://www.oracle.com/database/technologies/instant-client.html) [LINUX/WINDOWS]

`UBUNTU`
```bash
$ wget [oracle client instance file path]
$ sudo apt install alien
$ sudo alien -i [file.rpm]
```

- https://www.oracle.com/database/technologies/appdev/python/quickstartpythononprem.html

```bash
$ python -m pip install cx_Oracle --upgrade --user
```
```python

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

### DUAL
```sql
SELECT [operation] FROM DUAL;
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
SELECT [base column], [aggregation]([target column]) FROM GROUP BY [base column] HAVING [condition for aggregration base column];
```
`example`
```sql
SELECT [column1], count([column2]) FROM GROUP BY [column1] HAVING [condition for aggregation column1];
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

### VARCHAR
```sql
SELECT LENGTH('ABC') FROM dual;
```

```sql
SELECT substr('123456789', 7) FROM dual;   -- '789'
SELECT substr('123456789', -3) FROM dual;  -- '789'
SELECT substr('123456789', 7,3) FROM dual; -- '789'
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
### Integrity
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

### Binning
```sql
```

### Statistics
```sql
```

---

## Database Basic Concept

### Transaction
- Automicity
- Consistency
- Isolation
- Durability



