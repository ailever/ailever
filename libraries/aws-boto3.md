## [Cloud Computing] AWS-Boto3 | [Docs](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html#) | [GitHub]() | [PyPI]()


- [S3 Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
- [Athena Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/athena.html)
- [SageMaker Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html)

---

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

`Delete the S3 bucket`
```python
import boto3

s3_client = boto3.client('s3')
s3_client.delete_bucket(Bucket='ailever-sagemaker-bucket')
```
