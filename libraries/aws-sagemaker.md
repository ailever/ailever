## [Cloud Computing] AWS-SageMaker | [Docs](https://sagemaker.readthedocs.io/en/stable/index.html) | [GitHub](https://github.com/aws/sagemaker-python-sdk) | [PyPI]()


- https://aws.amazon.com/sagemaker/
- https://aws.amazon.com/sagemaker/studio/
- https://github.com/aws/amazon-sagemaker-examples

---

<br><br><br>


## SageMaker Notebook Instance
`SageMaker FileSystem`
```
|-- home
|   |-- ec2-user
|   |   |-- SageMaker

```

`Region & Bucket`
```python
import sagemaker

sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = sess.default_bucket()

print('REGION:', region)
print('BUCKET:', bucket)
```


## SageMaker Studio





---
