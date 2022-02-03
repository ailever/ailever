## Adult

### Exploratory Data Analysis

```python
from ailever.dataset import UCI
from ailever.analysis import EDA

frame = UCI.adult(download=False)
eda = EDA(frame, verbose=False)
eda.frame
```
![image](https://user-images.githubusercontent.com/56889151/152403742-9ab7cd0b-f271-48fd-9fc4-779ba0cfbc1c.png)

```python
display(eda.table_definition())
display(eda.attributes_specification())
eda.frame.info()
```
![image](https://user-images.githubusercontent.com/56889151/152403838-8b4ba316-c731-46cc-8144-08883687b2bd.png)


