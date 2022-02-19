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
delete from user where user='[account]';
flush privileges;
```
```sql
delete from user where user='test_account';
flush privileges;
```

---

## Metatable
```sql
show databases;
show tables;
```

```sql
select user();
select current_user();
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
