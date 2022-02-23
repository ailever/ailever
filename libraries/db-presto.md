- https://github.com/prestodb/presto
- https://prestodb.io/
  - https://prestodb.io/download.html 
  - https://prestodb.io/docs/current/index.html

---

## Installation
`download`: presto-server-0.270.tar.gz, presto-cli-0.270-executable.jar  

`STEP1`
```
|-- /home/user/presto-server-0.270
    |-- NOTICE
    |-- README.txt
    |-- bin
        |-- presto-cli-0.270-executable.jar
    |-- lib
    |-- plugin
```
```bash
~ $ tar -zxvf presto-server-0.270.tar.gz
~ $ mv presto-cli-0.270-executable.jar /home/user/presto-server-0.270/bin/
```
`STEP2`
```
|-- /home/user/presto-server-0.270
    |-- NOTICE
    |-- README.txt
    |-- bin
        |-- presto # rename 'presto-cli-0.270-executable.jar' to 'presto' 
    |-- lib
    |-- plugin
```
```bash
~/presto-server-0.270/bin $ mv presto-cli-0.270-executable.jar presto
~/presto-server-0.270/bin $ chmod +x presto
```

