- [Data types](https://docs.oracle.com/cd/A58617_01/server.804/a58241/ch5.htm)

## Metatable
### Table Information
```sql
SELECT * FROM all_tables WHERE owner = '[username]';
SELECT * FROM user_tables;
SELECT * FROM tabs;
SELECT * FROM cols;
```


## Syntax
### Create
```sql
CREATE TABLE [new_table] AS (SELECT * FROM [origin_table]);      -- copy table
```

### Update
```
ALTER TABLE [table] ADD [new_column] [data-type] DEFAULT [default-value] NOT NULL;    -- add column
ALTER TABLE [table] DROP COLUMN [column];                                             -- delete column
```

