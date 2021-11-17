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

s3_client = boto3.client('s3') # Retrieve the list of existing buckets
response = s3_client.list_buckets()
response['Buckets']
```

`Create an Amazon S3 bucket`
```python
import boto3

sess = boto3.Session()
region = sess.region_name

s3_client = boto3.client('s3', region_name=region)
s3_client.create_bucket(Bucket='ailever-sagemaker-bucket', CreateBucketConfiguration=dict(LocationConstraint=region))
```
