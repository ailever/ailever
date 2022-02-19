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
```

---


## Account Management
### Login to mysql server
```bash
$ sudo mysql -u root
```

### User inquery
```sql
use mysql;
select user, host from user;
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
