## [Cloud Computing] AWS-SageMaker | [Docs](https://sagemaker.readthedocs.io/en/stable/index.html) | [GitHub](https://github.com/aws/sagemaker-python-sdk) | [PyPI]()


- https://aws.amazon.com/sagemaker/
- https://aws.amazon.com/sagemaker/studio/
- https://github.com/aws/amazon-sagemaker-examples

---

<br><br><br>


## Amazon SageMaker Notebook Instance
`SageMaker FileSystem(EBS)` : /home/ec2-user/**SageMaker** 

```
|-- home
|   |-- ec2-user
|   |   |-- SageMaker

```

`Region & Bucket`
```python
import sagemaker

role = sagemaker.get_execution_role()
sess = sagemaker.Session()
region = sess.boto_session.region_name
# import boto3
# region = boto3.Session().region_name


# S3 bucket for saving code and model artifacts. Feel free to specify a different bucket and prefix
default_bucket = sagemaker.Session().default_bucket()
default_bucket_path = f"s3://{default_bucket}"

print('ROLE :', role)
print('REGION :', region)
print('BUCKET_PATH :', output_bucket_path)
```


### Amazon SageMaker Studio



## Amazon SageMaker MLOps
### Feature Store
### Model Registry


## Amazon SageMaker Workflow



---
