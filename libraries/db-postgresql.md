## Listener Check
`LINUX`
```bash
$ ps aux
$ netstat -nlpt
$ sudo service --status-all
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
$ sudo apt update
$ sudo apt install postgresql postgresql-contrib
$ sudo service --status-all
$ sudo service postgresql start
```

---

## Metatable

`SHOW DATABASES`
```sql
SELECT * FROM pg_catalog.pg_tables;
```


---

## Syntax

---
