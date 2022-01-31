## Tensorflow Dataset
- https://www.tensorflow.org/guide/data
- https://www.tensorflow.org/guide/data_performance
- https://www.tensorflow.org/guide/data_performance_analysis
- https://www.tensorflow.org/text
- https://www.tensorflow.org/tutorials
- https://github.com/ailever/ailever/tree/master/ailever/dataset

### Tensorflow I/O

### Tensorflow Record

<br><br><br>

---


## Tabular Dataset
- https://www.tensorflow.org/tutorials/structured_data/feature_columns

### From Dataframe
```python
from ailever.dataset import UCI
from ailever.analysis import EDA

dataset = UCI.adult(download=False)
eda = EDA(dataset, verbose=False)
dataset = eda.cleaning(as_str=all)

iterable_dataset = tf.data.Dataset.from_tensor_slices(dict(dataset))
for dictionary in iterable_dataset.batch(4).take(1):
    # dictionary ~ row-batch in dataframe
    for key, value in dictionary.items():
        # keys(column names), values(instances) in row-batch from dataframe
        print(key, value)
```


### From .CSV
```python
import tensorflow as tf
from ailever.dataset import UCI
from ailever.analysis import EDA

eda = EDA(UCI.adult(download=False), verbose=False) 
eda.cleaning(as_str=all).to_csv('adult.csv', index=False)

iterable_dataset = tf.data.experimental.make_csv_dataset('adult.csv', batch_size=4, label_name="50K")
for ordered_dictionary, tf_target in iterable_dataset.take(1):
    # ordered_dictionary ~ row-batch in *.csv
    for key, value in ordered_dictionary.items():
        # keys(column names), values(instances) in row-batch from *.csv
        print(key, value)
```

```python
import tensorflow as tf
from ailever.dataset import UCI
from ailever.analysis import EDA

eda = EDA(UCI.adult(download=False), verbose=False) 
eda.cleaning(as_str=all).to_csv('adult.csv', index=False)

iterable_dataset = tf.data.experimental.make_csv_dataset('adult.csv', batch_size=4, label_name="50K", select_columns=['50K', 'age', 'education-num', 'hours-per-week'])
for ordered_dictionary, tf_target in iterable_dataset.take(1):
    # ordered_dictionary ~ row-batch in *.csv
    for key, value in ordered_dictionary.items():
        # keys(column names), values(instances) in row-batch from *.csv
        print(key, value)
```


<br><br><br>

---


## Text Dataset

<br><br><br>

---


## Audio Dataset

<br><br><br>

---


## Image Dataset

<br><br><br>

---


## Video Dataset

<br><br><br>

---




