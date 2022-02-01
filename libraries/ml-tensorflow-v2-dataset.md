## Tensorflow Dataset
- https://www.tensorflow.org/guide/data
- https://www.tensorflow.org/guide/data_performance
- https://www.tensorflow.org/guide/data_performance_analysis
- https://www.tensorflow.org/text
- https://www.tensorflow.org/tutorials
- https://github.com/ailever/ailever/tree/master/ailever/dataset

### Tensorflow I/O

### Tensorflow Record


---


<br><br><br>
## Tabular Dataset
- https://www.tensorflow.org/tutorials/structured_data/feature_columns

### TF-API: Loader for Dataframe
`tf.data.Dataset.from_tensor_slices`
```python
import tensorflow as tf
from ailever.dataset import UCI
from ailever.analysis import EDA

dataset = UCI.adult(download=False)
eda = EDA(dataset, verbose=False)
dataset = eda.cleaning(as_str=all)

iterable_dataset = tf.data.Dataset.from_tensor_slices(dict(dataset))
for dictionary_batch in iterable_dataset.batch(4).take(1):
    # dictionary_batch ~ row-batch in dataframe
    for key, tf_value in dictionary_batch.items():
        # keys(column names), tf_value(instances_by_each_column) in row-batch from dataframe
        print(key, tf_value)
```

#### Dataframe-Pipelining
```python
```

---

### TF-API: Loader for .CSV
`tf.data.experimental.make_csv_dataset`
```python
import tensorflow as tf
from ailever.dataset import UCI
from ailever.analysis import EDA

eda = EDA(UCI.adult(download=False), verbose=False) 
eda.cleaning(as_str=all).to_csv('adult.csv', index=False)

iterable_dataset = tf.data.experimental.make_csv_dataset('adult.csv', num_epochs=1, batch_size=4, label_name="50K")
for ordered_dictionary_batch, tf_target in iterable_dataset.take(1):
    # ordered_dictionary_batch ~ row-batch in *.csv
    for key, tf_value in ordered_dictionary_batch.items():
        # keys(column names), tf_value(instances_by_each_column) in row-batch from *.csv
        print(key, tf_value)
```

```python
import tensorflow as tf
from ailever.dataset import UCI
from ailever.analysis import EDA

eda = EDA(UCI.adult(download=False), verbose=False) 
eda.cleaning(as_str=all).to_csv('adult.csv', index=False)

iterable_dataset = tf.data.experimental.make_csv_dataset('adult.csv', num_epochs=1, batch_size=4, label_name="50K", select_columns=['50K', 'age', 'education-num', 'hours-per-week'])
for ordered_dictionary_batch, tf_target in iterable_dataset.take(1):
    # ordered_dictionary_batch ~ row-batch in *.csv
    for key, tf_value in ordered_dictionary_batch.items():
        # keys(column names), tf_value(instances_by_each_column) in row-batch from *.csv
        print(key, tf_value)
```

`tf.data.experimental.CsvDataset`
```python
import tensorflow as tf
from ailever.dataset import UCI
from ailever.analysis import EDA

eda = EDA(UCI.adult(download=False), verbose=False) 
eda.cleaning(as_str=all).to_csv('adult.csv', index=False)

data_types  = [tf.int32, tf.string, tf.string, tf.string, tf.string, tf.string, tf.string, tf.string, tf.string, tf.string, tf.string, tf.string, tf.string, tf.string, tf.string] 
iterable_dataset = tf.data.experimental.CsvDataset('adult.csv', data_types, header=True)
for tuple_batch in iterable_dataset.batch(4).take(1):
    # tuple_batch ~ row-batch in *.csv
    for tf_value in tuple_batch:
        # tf_value(instances_by_each_column) in row-batch from *.csv
        print(tf_value)
```
```python
import tensorflow as tf
from ailever.dataset import UCI
from ailever.analysis import EDA

eda = EDA(UCI.adult(download=False), verbose=False) 
eda.cleaning(as_str=all).to_csv('adult.csv', index=False)

data_types  = [tf.int32, tf.string, tf.string, tf.string, tf.string] 
iterable_dataset = tf.data.experimental.CsvDataset('adult.csv', data_types, header=True, select_cols=[0,1,2,3,4])
for tuple_batch in iterable_dataset.batch(4).take(1):
    # tuple_batch ~ row-batch in *.csv
    for tf_value in tuple_batch:
        # tf_value(instances_by_each_column) in row-batch from *.csv
        print(tf_value)
```

```python
import tensorflow as tf
from ailever.dataset import UCI
from ailever.analysis import EDA

eda = EDA(UCI.adult(download=False), verbose=False) 
eda.cleaning(as_str=all).to_csv('adult.csv', index=False)

data_types  = [tf.int32, tf.string, tf.string] 
iterable_dataset = tf.data.experimental.CsvDataset('adult.csv', data_types, header=True, select_cols=[0,1,2])
iterable_dataset = iterable_dataset.map(lambda x0, x1, x2: (tf.cast(x0, tf.float32), x1, x2))
for tuple_batch in iterable_dataset.batch(4).take(1):
    # tuple_batch ~ row-batch in *.csv
    for tf_value in tuple_batch:
        # tf_value(instances_by_each_column) in row-batch from *.csv
        print(tf_value)
```

`tf.data.TextLineDataset`
```python
import tensorflow as tf
from ailever.dataset import UCI
from ailever.analysis import EDA

eda = EDA(UCI.adult(download=False), verbose=False) 
eda.cleaning(as_str=all).to_csv('adult.csv', index=False)

iterable_dataset = tf.data.TextLineDataset('adult.csv')
for tf_batch in iterable_dataset.batch(4).take(1):
    print(tf_batch)
```


#### .CSV-Pipelining
```python
import tensorflow as tf
import time

class CustomDataset(tf.data.Dataset):
    def _generator(num_samples):
        # 파일 열기
        time.sleep(0.03)

        for sample_idx in range(num_samples):
            # 파일에서 데이터(줄, 기록) 읽기
            time.sleep(0.015)
            yield (sample_idx,)

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=tf.dtypes.int64,
            output_shapes=(1,),
            args=(num_samples,)
        )

def benchmark(dataset, name, num_epochs=2):
    start_time = time.perf_counter()
    for _ in tf.data.Dataset.range(num_epochs):
        for _ in dataset:
            #time.sleep(0.01)            
            pass
    tf.print(f"%-{50}s"%f"실행 시간[{name}]", time.perf_counter() - start_time)
    
def mapped_function(x):
    # Do some hard pre-processing
    tf.py_function(lambda: time.sleep(0.003), [], ())
    return x+1

def fast_mapped_function(x):
    return x+1

# The naive approach
benchmark(CustomDataset(), name='The naive approach')

# Prefetching
benchmark(CustomDataset().prefetch(tf.data.experimental.AUTOTUNE), name='Prefetching')

# Parallelizing data extraction
benchmark(tf.data.Dataset.range(2).interleave(CustomDataset), name='Sequential interleave')
benchmark(tf.data.Dataset.range(2).interleave(CustomDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE), name='Parallel interleave')

# Parallelizing data transformation
benchmark(CustomDataset().map(mapped_function), name='Sequential mapping')
benchmark(CustomDataset().map(mapped_function, num_parallel_calls=tf.data.experimental.AUTOTUNE), name='Parallel mapping')

# Caching
benchmark(CustomDataset().map(mapped_function).cache(), num_epochs=5, name='Caching')

# Vectorizing mapping
benchmark(tf.data.Dataset.range(1000).map(mapped_function).batch(256), name='Hard Pre-Processing) Scalar mapping')
benchmark(tf.data.Dataset.range(1000).batch(256).map(mapped_function), name='Hard Pre-Processing) Vectorizing mapping')
benchmark(tf.data.Dataset.range(1000).map(fast_mapped_function).batch(256), name='Soft Pre-Processing) Scalar mapping')
benchmark(tf.data.Dataset.range(1000).batch(256).map(fast_mapped_function), name='Soft Pre-Processing) Vectorizing mapping')
```

---


<br><br><br>
## Text Dataset
### TF-API: Loader for .TXT
```python
```
#### TXT-Pipelining
```python
```

<br><br><br>

---


<br><br><br>
## Audio Dataset
### TF-API: Loader for .MP3
```python
```
#### .MP3-Pipelining
```python
```

---


<br><br><br>
## Image Dataset
### TF-API: Loader for .PNG
```python
```
#### .PNG-Pipelining
```python
```

---


<br><br><br>
## Video Dataset
### TF-API: Loader for .MP4
```python
```
#### .MP4-Pipelining
```python
```


---




