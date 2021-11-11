- https://aws.amazon.com/sagemaker/
- https://aws.amazon.com/sagemaker/studio/
- https://github.com/aws/sagemaker-python-sdk
- https://github.com/aws/amazon-sagemaker-examples


---

<br><br><br>


```python
import sagemaker

sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = sess.default_bucket()

print('REGION:', region)
print('BUCKET:', bucket)
```



---
