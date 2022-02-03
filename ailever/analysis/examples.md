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
![image](https://user-images.githubusercontent.com/56889151/152404253-fb41f0da-1d52-4f14-be3c-1446eca893af.png)
![image](https://user-images.githubusercontent.com/56889151/152404343-c35a844d-e391-4e5f-b9ec-be0eeb4c741d.png)
![image](https://user-images.githubusercontent.com/56889151/152404380-9a9cb3e8-2223-410b-a4a0-bd841c778b45.png)


