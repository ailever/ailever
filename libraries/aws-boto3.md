## [Cloud Computing] AWS- | [Docs]() | [GitHub]() | [PyPI]()

```python
import boto3

sess = boto3.Session()
region = sess.region_name
print('REGION :', region)
```

`List existing buckets`
```python
import boto3

# Retrieve the list of existing buckets
s3 = boto3.client('s3')
response = s3.list_buckets()

# Output the bucket names
print('Existing buckets:')
for bucket in response['Buckets']:
    print(f'  {bucket["Name"]}')
```
