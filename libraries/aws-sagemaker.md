## [Cloud Computing] | [aws-sagemaker](https://sagemaker.readthedocs.io/en/stable/index.html) | [github](https://github.com/aws/sagemaker-python-sdk)

- https://aws.amazon.com/sagemaker/
- https://aws.amazon.com/sagemaker/studio/
- https://github.com/aws/amazon-sagemaker-examples

---

<br><br><br>


`Region & Bucket`
```python
import sagemaker

sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = sess.default_bucket()

print('REGION:', region)
print('BUCKET:', bucket)
```



---
