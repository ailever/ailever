## [Cloud Computing] AWS-Boto3 | [Docs](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html#) | [GitHub]() | [PyPI]()


- [S3 Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html)
- [Athena Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/athena.html)
- [SageMaker Client](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html)

---

## Boto3: The AWS SDK for Python
```python
import boto3

sess = boto3.Session()
region = sess.region_name
print('REGION :', region)
```

## Athena 
### Athena Client

## S3
### S3 Client
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
s3_client.create_bucket(Bucket='ailever-bucket', CreateBucketConfiguration=dict(LocationConstraint=region))
```

`Delete the S3 bucket`
```python
import boto3

s3_client = boto3.client('s3')
s3_client.delete_bucket(Bucket='ailever-bucket')
```

`Upload/Download a file to/from an S3 object`
```python
from ailever.dataset import SKAPI
import boto3

# download dataset
SKAPI.digits(download=True)

s3_client = boto3.client("s3")
s3_client.upload_file(Bucket='ailever-bucket', Key='dataset/digits.csv', Filename='digits.csv') # s3://ailever-bucket/dataset/digits.csv
s3_client.download_file(Bucket='ailever-bucket', Key='dataset/digits.csv', Filename='digits.csv') # s3://ailever-bucket/dataset/digits.csv
```

### S3 Resource
`List existing objects on a folder in the S3 Bucket`
```python
import boto3

s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket('ailever-bucket')
list(map(lambda x: x.key, bucket.objects.filter(Prefix="folder/")))
```
`Delete a folder in the S3 Bucket`
```python
import boto3

s3_resource = boto3.resource('s3')
bucket = s3_resource.Bucket('ailever-bucket')
bucket.objects.filter(Prefix="folder/").delete() # s3://ailever-bucket/folder/
```
`Delete a file in the S3 Bucket`
```python
import boto3

s3_resource = boto3.resource('s3')
s3_resource.Object('ailever-bucket', 'file').delete() # s3://ailever-bucket/file
```

## SageMaker Client

