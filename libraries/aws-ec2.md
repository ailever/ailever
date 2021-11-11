- https://aws.amazon.com/
- https://console.aws.amazon.com/


`ACCESS TO EC2 INSTANCE`

```bash
$ chmod 400 keypair_client.pem
$ ssh -v -i keypair_client.pem ec2-user@11.22.33.44
```



`Security Credentials`  
- https://console.aws.amazon.com/ > Security Credentials > Create Access Key
```python
AWS_ACCESS_KEY_ID = 'AWS_ACCESS_KEY_ID'
AWS_SECRET_ACCESS_KEY = 'AWS_SECRET_ACCESS_KEY'
MERCHANT_ID = 'MERCHANT_ID'
MARKETPLACE_ID = 'MARKETPLACE_ID' 
```
