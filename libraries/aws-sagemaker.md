## [Cloud Computing] AWS-SageMaker | [Docs](https://sagemaker.readthedocs.io/en/stable/index.html) | [GitHub](https://github.com/aws/sagemaker-python-sdk) | [PyPI]()


- https://aws.amazon.com/sagemaker/
- https://aws.amazon.com/sagemaker/studio/
- https://github.com/aws/amazon-sagemaker-examples
- https://docs.aws.amazon.com/sagemaker/index.html

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
#### Scikit-Learn with the SageMaker Python SDK
`script`
```python
```
`main`
```python
```

#### XGBoost with the SageMaker Python SDK
`script`
```python
```
`main`
```python
```

#### Pytorch with the SageMaker Python SDK
`script`
```python
```
`main`
```python
```

#### Tensorflow with the SageMaker Python SDK
`script`
```python
```
`main`
```python
```




---

## Amazon SageMaker Resource for MLOps
### Amazon SageMaker Project
#### Templates
![image](https://user-images.githubusercontent.com/56889151/143676034-9d169093-b32d-422c-a2fa-ce9497b2465e.png)


### Pipelines
![image](https://user-images.githubusercontent.com/56889151/143676158-841d4048-8786-4526-b9a8-66a9dd9b1f55.png)

#### UI Configuration
- Executions
- Graph
- Parameters
- Settings
  - Metadata
    - [Pipeline Arn] arn:aws:sagemaker:{Regsion}:{?}:pipeline/{ProjectName}-{ProjectID}
    - [Role Arn] arn:aws:iam:{?}:role/service-role/AmazonSageMakerServiceCatalogProductsUseRole
    - [Creation Time] 2021-11-27T06:43:48.000Z
    - [Last Modified Time] 2021-11-27T06:55:13.000Z
    - [Tags] sagemaker:project-name: {ProjectName}, sagemaker:project-id: {ProjectID}

#### ProcessingStep, TrainingStep, ConditionStep
```python
```


### Model Registry
#### UI Configuration
- Versions
  - Activity
  - Model Quality
  - Explainability
  - Bias report
  - Settings
    - Metadata
      - [Project]
      - [Pipeline]
      - [Execution]
      - [Model Group]
      - [ECR image URI]
      - [Model location(S3)]
      - [Modified on]
      - [Modified by]
      - [Created on]
      - [Created by]
      - [ARN]
      - [Trial Component]
- Settings


### Data Wrangler
### Experiments and trials
#### UI Configuration
- Trials/Trial Components
  - [Charts]
  - [Metrics]
  - [Parameters]
  - [Artifacts]
    - Input Artifacts: SageMaker.ImageUri, code, input-1(output/model.tar.gz), input-2(output/test) 
    - Output Artifacts: evaluation(output/evaluation)
  - [AWS Settings]
    - Job Settings: Job name, ARN, Creation time, Processing start time, Processing end time, Processing duration(seconds)
    - Algorithm: Processing image, Instance type, Instance count, Volumne size in GB, Volumne KMS Key id
    - Job Exit: Failure reason, Exit message
    - Monitor: View logs(CloudWatch)
  - [Debugger]
  - [Explainability]
  - [Bias report]
  - [Trial Mappings]


### Endpoints
### Compilation Jobs
### Feature Store


---

## Amazon SageMaker Workflow



---
