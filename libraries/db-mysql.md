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

### CREATE
```sql
CREATE TABLE TABLE01 (
    COL1 int,
    COL2 varchar(255),
    COL3 varchar(255),
    COL4 varchar(255),
    COL5 varchar(255)
);
```


### INSERT INTO
```sql
INSERT INTO TABLE01
VALUES (1, 'col2_instance', 'col3_instance', 'col4_instance', 'col5_instance');
```
```sql
INSERT INTO TABLE01 (COL2, COL3)
VALUES ('col2_instance', 'col3_instance');
```

### SELECT FROM

### DELETE

### ALTER


---

## Data Analysis



---

## Reference
