- [Data types](https://docs.oracle.com/cd/A58617_01/server.804/a58241/ch5.htm)
- [Statistics](https://www.oracle.com/database/technologies/bi-datawarehousing.html)

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
Task Order: FORM > WHERE > GROUP BY > HAVING > SELECT > ORDER BY 

### SELECT
```sql
SELECT * FROM [table];
SELECT table1.*, table2.* FROM [table1], [table2];
```

#### SELECT CASE WHEN THEN FROM
```sql
```

#### SELECT FROM WHERE LIKE
```sql
```


#### SELECT FROM GROUP BY HAVING
```sql
SELECT [column], [aggregation]([column]) FROM GROUP BY [column] HAVING [condition for aggregation column];
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

### DROP
```sql
DROP TABLE [table];
```


### WITH
```sql
WITH [new_table] AS (SELECT * FROM [origin_table])
SELECT * FROM [new_table];
```
```sql
WITH [new_table1] AS (SELECT * FROM [origin_table1]),
     [new_table2] AS (SELECT * FROM [origin_table2]),
     [new_table3] AS (SELECT * FROM [new_table1]),
     [new_table3] AS (SELECT * FROM [new_table2]),
     [final_table] AS (SELECT * FROM [new_table3], [new_table4]),     
SELECT * FROM [final_table] ;
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
```sql
```

### UNION
```sql
SELECT * FROM [table1]
UNION ALL
SELECT * FROM [table2];
```

## Column Control
### ROW_NUMBER()
```sql
```

## Row Control
### INSERT
```sql
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

### VARCHAR
```sql
```

### NUMBER
```sql
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
WITH nullcounting AS (SELECT count([column]) rnefnv, count(*) rn FROM [table])
SELECT rn - rnefnv FROM nullcounting;        -- rn: rownum, rnefnv: rownum_except_for_null_values
```

```sql
SELECT count(*) FROM [table] WHERE [column] IS NULL;
```


### Cleaning
```sql
SELECT * FROM [table] WHERE [column] IS NOT NULL;
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



