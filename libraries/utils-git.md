## Git Syntax

```bash
$ ssh-keygen -t rsa -b 4096
$ ssh-keygen -t dsa
$ ssh-keygen -t ecdsa -b 521
$ ssh-keygen -t ed25519
```

### Config

`Config Syntax`
```bash
$ git config [--global or --local] [category].[parameter] [value]
```

`list`
```bash
$ git config --list
```


#### Global Config
`~/.gitconfig`  
```bash
$ git config --global user.name "user"
$ git config --global user.email "user@domain"
```



#### Local Config
`./.git/config`



### remote

```bash
$ git remote -v
```

```bash
$ git remote rm origin
$ git remote add origin [https://*/*.git]
$ git push --set-upstream origin master
```

```bash
$ git remote add upstream [git repository address]
```

---

### push

```bash
$ git push upstream
```

<br><br><br>

--- 


## Usecase
