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





---

## Amazon SageMaker Resource for MLOps
### Amazon SageMaker Project
create project > Model Registry

### Pipelines
Metadata
- [Pipeline Arn] arn:aws:sagemaker:{Regsion}:{?}:pipeline/{ProjectName}-{ProjectID}
- [Role Arn] arn:aws:iam:{?}:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole
- [Creation Time] 2021-11-27T06:43:48.000Z
- [Last Modified Time] 2021-11-27T06:55:13.000Z
- [Tags] sagemaker:project-name: {ProjectName}, sagemaker:project-id: {ProjectID}

### Model Registry
### Data Wrangler
### Experiments and trials
### Endpoints
### Compilation Jobs
### Feature Store


---

## Amazon SageMaker Workflow



---
