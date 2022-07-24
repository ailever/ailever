## [Cloud Computing] AWS-EC2 | [Docs]() | [GitHub]() | [PyPI]()

- https://aws.amazon.com/
- https://console.aws.amazon.com/


---

<br><br><br>

`ACCESS TO EC2 INSTANCE`

```bash
$ chmod 400 keypair_client.pem
$ ssh -v -i keypair_client.pem ec2-user@11.22.33.44
```



`Security Credentials`: https://console.aws.amazon.com/ > Security Credentials > Create Access Key
```python
AWS_ACCESS_KEY_ID = 'AWS_ACCESS_KEY_ID'
AWS_SECRET_ACCESS_KEY = 'AWS_SECRET_ACCESS_KEY'
MERCHANT_ID = 'MERCHANT_ID'
MARKETPLACE_ID = 'MARKETPLACE_ID' 
```


### DevOps
#### Jira
```bash
$ sudo yum update
$ sudo yum install docker-io
$ sudo systemctl start docker
$ sudo setfacl -m user:ec2-user:rw /var/run/docker.sock
```

```bash
$ docker rm --volumes --force "jira-container"                      # delete container
$ docker pull cptactionhank/atlassian-jira-software:latest          # download docker image
$ docker create --restart=no --name "jira-container" --publish "8080:8080" --volume "hostpath:/var/atlassian/jira" --env "CATALINA_OPTS= -Xms1024m -Xmx1024m -Datlassian.plugins.enable.wait=300" cptactionhank/atlassian-jira-software:latest
$ docker start --attach "jira-container"                            # start docker container
```

#### Confluence
```bash
```


