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
select user, host, authentication_string, plugin from user;
```

### Add user
`create user`
```sql
create user '[account]'@'[ip]' identified by '[passwd]';
```
```sql
create user 'test'@'localhost' identified by 'passwd';
create user 'test'@'%' identified by 'passwd';
```

`grant all privileges on`
```sql
grant all privileges on [dbname].* to [account]@'[ip]' identified by '[passwd]';
```
```sql
grant all privileges on samdb01.* to test_account@'localhost' identified by 'passwd';
grant all privileges on samdb01.* to test_account@'%' identified by 'passwd';
```



### Privileges
```sql
grant all privileges on [dbname].* to [account]@'[ip]';
```
```sql
grant all privileges on samdb01.* to test_account@'localhost';
grant all privileges on samdb01.* to test_account@'%';
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
