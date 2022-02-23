- https://github.com/prestodb/presto
- https://prestodb.io/
  - https://prestodb.io/download.html 
  - https://prestodb.io/docs/current/index.html

---

## Installation
### Download
`download`:  
- presto-server-0.270.tar.gz
- presto-cli-0.270-executable.jar 
- presto-jdbc-0.270.jar 

### STEP1
```
|-- /home/user/presto-server-0.270
    |-- NOTICE
    |-- README.txt
    |-- bin
        |-- presto-cli-0.270-executable.jar
        |-- presto-jdbc-0.270.jar 
        |-- launcher  
        |-- launcher.properties  
        |-- launcher.py
    |-- lib
    |-- plugin
```
```bash
~ $ tar -zxvf presto-server-0.270.tar.gz
~ $ mv presto-cli-0.270-executable.jar /home/user/presto-server-0.270/bin/
~ $ mv presto-jdbc-0.270.jar /home/user/presto-server-0.270/bin/
```

### STEP2
```
|-- /home/user/presto-server-0.270
    |-- NOTICE
    |-- README.txt
    |-- bin
        |-- presto # rename 'presto-cli-0.270-executable.jar' to 'presto' 
        |-- launcher  
        |-- launcher.properties  
        |-- launcher.py
    |-- lib
    |-- plugin
    |-- prestodata
    |-- etc
        |-- config.properties
        |-- jvm.config
        |-- log.properties
```
```bash
~/presto-server-0.270 $ mv bin/presto-cli-0.270-executable.jar bin/presto
~/presto-server-0.270 $ chmod +x bin/presto
~/presto-server-0.270 $ mkdir prestodata
~/presto-server-0.270 $ mkdir etc
~/presto-server-0.270 $ touch etc/config.properties
~/presto-server-0.270 $ touch etc/jvm.config
~/presto-server-0.270 $ touch etc/log.properties
```

### STEP3  
- https://prestodb.io/docs/current/installation/deployment.html
`config.properties`  
```
coordinator=true
node-scheduler.include-coordinator=true
http-server.http.port=8080
query.max-memory=10GB
query.max-memory-per-node=1GB
discovery-server.enabled=true
discovery.uri=http://127.0.0.1:8080
```
`jvm.config`
```
-server
-Xmx16G
-XX:+UseG1GC
-XX:G1HeapRegionSize=32M
-XX:+UseGCOverheadLimit
-XX:+ExplicitGCInvokesConcurrent
-XX:+HeapDumpOnOutOfMemoryError
-XX:+ExitOnOutOfMemoryError
```
`log.properties`
```
com.facebook.presto=INFO
```
`node.properties`
```
node.environment=production
node.id=f7c4bf3c-dbb4-4807-baae-9b7e41807bc9
node.data-dir=/home/user/presto/prestodata  # caution: path
```

### STEP4
`connector`  
- https://prestodb.io/docs/current/connector.html
- https://prestodb.io/docs/current/connector/mysql.html
```
|-- /home/user/presto-server-0.270
    |-- NOTICE
    |-- README.txt
    |-- bin
        |-- presto # rename 'presto-cli-0.270-executable.jar' to 'presto' 
        |-- launcher  
        |-- launcher.properties  
        |-- launcher.py
    |-- lib
    |-- plugin
    |-- prestodata
    |-- etc
        |-- config.properties
        |-- jvm.config
        |-- log.properties
        |-- catalog
            |-- mysql.properties
        
```
```bash
~/presto-server-0.270 $ mkdir etc/catalog
~/presto-server-0.270 $ touch etc/catalog/mysql.properties
```
`mysql.properties`
```
connector.name=mysql
connection-url=jdbc:mysql://127.0.0.1:3306
connection-user=root
connection-password=secret
```

### Execution
```bash
~/presto-server-0.270/bin $ ./launcher start
~/presto-server-0.270/bin $ ./presto --server 127.0.0.1:8080
```
```sql
SHOW catalogs;               -- USE [catalog].[schema]
SHOW SCHEMAS FROM mysql;     -- USE [schema]
use mysql.samdb01;
```
---

## DBeaver
![image](https://user-images.githubusercontent.com/56889151/155350064-8cf17746-c861-4e1d-9bb8-1003c241056a.png)
![image](https://user-images.githubusercontent.com/56889151/155350154-cf7e57a9-8dd9-4097-9f7e-65621f2bcd6f.png)

--

## Syntax



