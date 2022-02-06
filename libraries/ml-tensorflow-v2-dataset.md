## Tensorflow Dataset
- https://www.tensorflow.org/guide/data
- https://www.tensorflow.org/guide/data_performance
- https://www.tensorflow.org/guide/data_performance_analysis
- https://www.tensorflow.org/text
- https://www.tensorflow.org/tutorials
- https://github.com/ailever/ailever/tree/master/ailever/dataset

### Tensorflow I/O

### Tensorflow Record

### Keras Data Preprocessing
#### Sequence

`TimeseriesGenerator`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/TimeseriesGenerator
```python
from tensorflow.keras import preprocessing
import numpy as np

seqeunce = np.array([[i] for i in range(50)])
iterable_dataset = preprocessing.sequence.TimeseriesGenerator(
    seqeunce, seqeunce, length=10, sampling_rate=1, batch_size=2)

batch = 0
X = iterable_dataset[batch][0] # return sequence
y = iterable_dataset[batch][1] # return target
X, y
```
```
(array([[[ 0],
         [ 1],
         [ 2],
         [ 3],
         [ 4],
         [ 5],
         [ 6],
         [ 7],
         [ 8],
         [ 9]],
 
        [[ 1],
         [ 2],
         [ 3],
         [ 4],
         [ 5],
         [ 6],
         [ 7],
         [ 8],
         [ 9],
         [10]]]),
 array([[10],
        [11]]))
```

`make_sampling_table`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/make_sampling_table
```python
```

`pad_sequences`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/pad_sequences
```python
from tensorflow.keras import preprocessing
import numpy as np

sequence = [[1], 
            [2, 3], 
            [4, 5, 6]]
preprocessing.sequence.pad_sequences(sequence, value=0, padding='post', maxlen=5) # padding: 'pre' or 'post'
```
```
array([[1, 0, 0, 0, 0],
       [2, 3, 0, 0, 0],
       [4, 5, 6, 0, 0]], dtype=int32)
```

`skipgrams`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/sequence/skipgrams
```python
```


#### Text
`texts_to_sequences`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer
```python
from tensorflow.keras import preprocessing

fitting_sentences = [
  'I love my dog',
  'I love my dog.',
  'I love my dog!',
  'I love my dog#',
  'I love my dog\n',
  'I love my dog\t',     
  'I love my dog\\',
  'I love my dog~']
tokenizer = preprocessing.text.Tokenizer(
    num_words=None,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, 
    split=' ', 
    char_level=False, 
    oov_token="<OOV>",
    document_count=0)
tokenizer.fit_on_texts(fitting_sentences)

sentences = ['I like my cat.', 
             'Cat like me!']
print(tokenizer.word_index)
print(tokenizer.texts_to_sequences(sentences))
```
```
{'<OOV>': 1, 'i': 2, 'love': 3, 'my': 4, 'dog': 5}
[[2, 1, 4, 1], [1, 1, 1]]
```


`text_to_word_sequence`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/text_to_word_sequence
```python
from tensorflow.keras import preprocessing

sample_text = 'This is a sample sentence.'
preprocessing.text.text_to_word_sequence(
    input_text=sample_text,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=' ')
```
```
['this', 'is', 'a', 'sample', 'sentence']
```

`hashing_trick`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/hashing_trick
```python
```

`one_hot`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/one_hot
```python
from tensorflow.keras import preprocessing

sample_text = 'This is a sample sentence.'
indices = preprocessing.text.one_hot(
    input_text=sample_text, 
    n=10,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True, split=' ')
indices
```
```
[1, 1, 7, 1, 8]
```

`tf.one_hot`
```python
import tensorflow as tf
from tensorflow.keras import preprocessing

pad_sequences = tf.constant([
    [1, 0, 0, 0, 0],
    [2, 3, 0, 0, 0],
    [4, 5, 6, 0, 0]], dtype=tf.int32)

num_unique_char = tf.unique(pad_sequences.numpy().flatten())[0].shape[0]  
onehot_sequences = tf.one_hot(pad_sequences, depth=num_unique_char) # depth=7
onehot_sequences
```
```
<tf.Tensor: shape=(3, 5, 7), dtype=float32, numpy=
array([[[0., 1., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.]],

       [[0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.]],

       [[0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 1.],
        [1., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.]]], dtype=float32)>
```

`tokenizer_from_json`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/tokenizer_from_json
```python
```



#### Image

`DirectoryIterator`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/DirectoryIterator
```python
```

`ImageDataGenerator`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
```python
```

`Iterator`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/Iterator
```python
```

`NumpyArrayIterator`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/NumpyArrayIterator
```python
```

`apply_affine_transform`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/apply_affine_transform
```python
```

`apply_brightness_shift`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/apply_brightness_shift
```python
```

`apply_channel_shift`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/apply_channel_shift
```python
```

`random_brightness`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_brightness
```python
```

`random_channel_shift`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_channel_shift
```python
```

`random_rotation`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_rotation
```python
```

`random_shear`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_shear
```python
```

`random_shift`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_shift
```python
```

`random_zoom`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/random_zoom
```python
```

`smart_resize`  
- https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/smart_resize
```python
```



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

#### Pipeline Optimization&Evaluation
```python
import time
import itertools
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

# Define Dataset
dataset = tf.random.normal(mean=100.0, stddev=9.0, shape=(100, 7)).numpy()
dataset = pd.DataFrame(dataset).add_prefix('COMP')

# Parameters for Parallelized Data Extraction
NUM_ROWS = dataset.shape[0]
BATCH_SIZE = 5
EPOCHS = 2

def draw_timeline(timeline, width=0.5, annotate=False, save=False):
    title = timeline["name"]
    
    # 타임라인에서 유효하지 않은 항목(음수 또는 빈 스텝) 제거
    invalid_mask = np.logical_and(timeline['timings'] > 0, timeline['steps'] != b'')[:,0]
    steps = timeline['steps'][invalid_mask].numpy()
    timings = timeline['timings'][invalid_mask].numpy()
    values = timeline['counters'][invalid_mask].numpy()

    # 처음 발견될 때 순서대로 다른 스텝을 가져옵니다.
    step_ids, indices = np.stack(np.unique(steps, return_index=True))
    step_ids = step_ids[np.argsort(indices)]

    # 시작 시간을 0으로 하고 최대 시간 값을 계산하십시오.
    min_time = timings[:,0].min()
    timings[:,0] = (timings[:,0] - min_time)
    end = max(width, (timings[:,0]+timings[:,1]).max() + 0.01)

    cmap = mpl.cm.get_cmap("plasma")
    plt.close()
    fig, axs = plt.subplots(len(step_ids), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle(title)
    fig.set_size_inches(25.0, len(step_ids))
    plt.xlim(-0.01, end)

    for i, step in enumerate(step_ids):
        step_name = step.decode()
        ax = axs[i]
        ax.set_ylabel(step_name)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("time (s)")
        ax.set_xticklabels([])
        ax.grid(which="both", axis="x", color="k", linestyle=":")

        # 주어진 단계에 대한 타이밍과 주석 얻기
        entries_mask = np.squeeze(steps==step)
        serie = np.unique(timings[entries_mask], axis=0)
        annotations = values[entries_mask]

        ax.broken_barh(serie, (0, 1), color=cmap(i / len(step_ids)), linewidth=1, alpha=0.66)
        if annotate:
            for j, (start, width) in enumerate(serie):
                annotation = "\n".join([f"{l}: {v}" for l,v in zip(("i", "e", "s"), annotations[j])])
                ax.text(start + 0.001 + (0.001 * (j % 2)), 0.55 - (0.1 * (j % 2)), annotation,
                        horizontalalignment='left', verticalalignment='center')
    if save:
        plt.savefig(title.lower().translate(str.maketrans(" ", "_")) + ".svg")

class CustomDataset(tf.data.Dataset):
    _BATCH_COUNTER = itertools.count()
    _EPOCHS_COUNTER = defaultdict(itertools.count)
    # OUTPUT: (indices, features, steps, timings, counters)    
    OUTPUT_TYPES = (tf.dtypes.int32, tf.dtypes.float32, tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int32)
    OUTPUT_SHAPES = ((4, ), (1, 7), (1, 1), (1, 2), (1, 3))
    
    def _generator(batch_idx, batch_size):
        epoch_idx = next(CustomDataset._EPOCHS_COUNTER[batch_idx])
        read_elapsed = time.perf_counter()
        for sample_idx, (row_idx, row_series) in enumerate(dataset.iloc[batch_idx*batch_size:(batch_idx+1)*batch_size].iterrows()):
            read_enter = time.perf_counter() - read_elapsed
            yield ([batch_idx, epoch_idx, sample_idx, row_idx], [row_series.values], [("Read",),], [(read_enter, read_elapsed),], [(batch_idx, epoch_idx, sample_idx)])
            read_enter = time.perf_counter()

    def __new__(cls, batch_size):
        return tf.data.Dataset.from_generator(
            cls._generator,
            args=(next(cls._BATCH_COUNTER), batch_size),
            output_types=cls.OUTPUT_TYPES,
            output_shapes=cls.OUTPUT_SHAPES)

def Extraction(*args, **kwargs):
    #tf.print("Data Extraction")
    return CustomDataset(BATCH_SIZE)

def tf_mapper(func):
    def wrapper(*args):
        return tf.py_function(func, inp=args, Tout=list(getattr(arg, 'dtype') for arg in args))
    return wrapper

@tf_mapper
def Preprocessing(indices, features, steps, timings, counters):
    #tf.print('Data Preprocessing')
    map_enter = time.perf_counter()    
    features = tf.linalg.normalize(features, axis=-1)[0]
    map_elapsed = time.perf_counter() - map_enter
    return indices, features, tf.concat((steps, [["Map"]]), axis=0), tf.concat((timings, [[map_enter, map_elapsed]]), axis=0), tf.concat((counters, [counters[-1]]), axis=0)


def IterableDataset_01(num_repeat=1):
    return tf.data.Dataset.range(num_repeat).interleave(Extraction, cycle_length=1).map(Preprocessing).batch(BATCH_SIZE, drop_remainder=True)

def benchmark(iterable_dataset, name):
    steps_acc = tf.zeros([0, 1], dtype=tf.dtypes.string)
    times_acc = tf.zeros([0, 2], dtype=tf.dtypes.float32)
    values_acc = tf.zeros([0, 3], dtype=tf.dtypes.int32)
    
    start_time = time.perf_counter()
    for batch_idx in range(NUM_ROWS//BATCH_SIZE):
        epoch_enter = time.perf_counter()        
        for indices, features, steps, timings, counters in iterable_dataset(num_repeat=EPOCHS):
            #steps = tf.squeeze(steps)
            #timings = tf.squeeze(timings)
            #counters = tf.squeeze(counters)
            steps = steps[0]
            timings = timings[0]
            counters = counters[0]
            
            # [dataset information]
            steps_acc = tf.concat((steps_acc, steps), axis=0)
            times_acc = tf.concat((times_acc, timings), axis=0)
            values_acc = tf.concat((values_acc, counters), axis=0)

            # [measure training time]
            train_enter = time.perf_counter()
            #print('%-12s'%f'[BATCHIDX:{batch_idx}] ')
            indices = pd.DataFrame(data=indices.numpy(), columns=['batch_idx', 'epoch_idx', 'sample_idx', 'row_idx'])
            features = pd.DataFrame(data=features.numpy().squeeze(), columns=dataset.columns)
            #display(pd.concat([indices, features], axis=1).set_index(['batch_idx', 'epoch_idx', 'sample_idx', 'row_idx']))

            # [training time information]
            train_elapsed = time.perf_counter() - train_enter
            steps_acc = tf.concat((steps_acc, [["Train"]]), axis=0)
            times_acc = tf.concat((times_acc, [(train_enter, train_elapsed)]), axis=0)
            values_acc = tf.concat((values_acc, [counters[-1]]), axis=0)

        # [epoch information]
        epoch_elapsed = time.perf_counter() - epoch_enter
        steps_acc = tf.concat((steps_acc, [["Epoch"]]), axis=0)
        times_acc = tf.concat((times_acc, [(epoch_enter, epoch_elapsed)]), axis=0)
        values_acc = tf.concat((values_acc, [[-1, batch_idx, -1]]), axis=0)
            
    exec_time = time.perf_counter() - start_time            
    tf.print(f"%-{90}s"%f"[처리 과정에 따른 실행 시간][{name}]", ':', exec_time)
    return {"name":name, "exec_time":exec_time, "steps": steps_acc, "timings": times_acc, "counters": values_acc}        
    
draw_timeline(benchmark(IterableDataset_01, name='Scalar Sequential Mapping'))
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

#### Pipeline Optimization&Evaluation
```python
import itertools
from collections import defaultdict
import time

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

def draw_timeline(timeline, title, width=0.5, annotate=False, save=False):
    # 타임라인에서 유효하지 않은 항목(음수 또는 빈 스텝) 제거
    invalid_mask = np.logical_and(timeline['times'] > 0, timeline['steps'] != b'')[:,0]
    steps = timeline['steps'][invalid_mask].numpy()
    times = timeline['times'][invalid_mask].numpy()
    values = timeline['values'][invalid_mask].numpy()

    # 처음 발견될 때 순서대로 다른 스텝을 가져옵니다.
    step_ids, indices = np.stack(np.unique(steps, return_index=True))
    step_ids = step_ids[np.argsort(indices)]

    # 시작 시간을 0으로 하고 최대 시간 값을 계산하십시오.
    min_time = times[:,0].min()
    times[:,0] = (times[:,0] - min_time)
    end = max(width, (times[:,0]+times[:,1]).max() + 0.01)

    cmap = mpl.cm.get_cmap("plasma")
    plt.close()
    fig, axs = plt.subplots(len(step_ids), sharex=True, gridspec_kw={'hspace': 0})
    fig.suptitle(title)
    fig.set_size_inches(25.0, len(step_ids))
    plt.xlim(-0.01, end)

    for i, step in enumerate(step_ids):
        step_name = step.decode()
        ax = axs[i]
        ax.set_ylabel(step_name)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("time (s)")
        ax.set_xticklabels([])
        ax.grid(which="both", axis="x", color="k", linestyle=":")

        # 주어진 단계에 대한 타이밍과 주석 얻기
        entries_mask = np.squeeze(steps==step)
        serie = np.unique(times[entries_mask], axis=0)
        annotations = values[entries_mask]

        ax.broken_barh(serie, (0, 1), color=cmap(i / len(step_ids)), linewidth=1, alpha=0.66)
        if annotate:
            for j, (start, width) in enumerate(serie):
                annotation = "\n".join([f"{l}: {v}" for l,v in zip(("i", "e", "s"), annotations[j])])
                ax.text(start + 0.001 + (0.001 * (j % 2)), 0.55 - (0.1 * (j % 2)), annotation,
                        horizontalalignment='left', verticalalignment='center')
    if save:
        plt.savefig(title.lower().translate(str.maketrans(" ", "_")) + ".svg")

class TimeMeasuredDataset(tf.data.Dataset):
    # 출력: (steps, timings, counters)
    OUTPUT_TYPES = (tf.dtypes.string, tf.dtypes.float32, tf.dtypes.int32)
    OUTPUT_SHAPES = ((2, 1), (2, 2), (2, 3))

    _INSTANCES_COUNTER = itertools.count()  # 생성된 데이터셋 수
    _EPOCHS_COUNTER = defaultdict(itertools.count)  # 각 데이터를 수행한 에포크 수

    def _generator(instance_idx, num_samples):
        epoch_idx = next(TimeMeasuredDataset._EPOCHS_COUNTER[instance_idx])

        # 파일 열기
        open_enter = time.perf_counter()
        time.sleep(0.03)
        open_elapsed = time.perf_counter() - open_enter

        for sample_idx in range(num_samples):
            # 파일에서 데이터(줄, 기록) 읽어오기
            read_enter = time.perf_counter()
            time.sleep(0.015)
            read_elapsed = time.perf_counter() - read_enter

            yield (
                [("Open",), ("Read",)],
                [(open_enter, open_elapsed), (read_enter, read_elapsed)],
                [(instance_idx, epoch_idx, -1), (instance_idx, epoch_idx, sample_idx)]
            )
            open_enter, open_elapsed = -1., -1.  # 음수는 필터링됨

    def __new__(cls, num_samples=3):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=cls.OUTPUT_TYPES,
            output_shapes=cls.OUTPUT_SHAPES,
            args=(next(cls._INSTANCES_COUNTER), num_samples)
        )

_batch_map_num_items = 10
def dataset_generator_fun(*args):
    print('extraction')
    return TimeMeasuredDataset(num_samples=_batch_map_num_items)    
    
def map_decorator(func):
    def wrapper(steps, times, values):
        # 자동 그래프가 메서드를 컴파일하지 못하도록 tf.py_function을 사용
        return tf.py_function(
            func,
            inp=(steps, times, values),
            Tout=(steps.dtype, times.dtype, values.dtype)
        )
    return wrapper

@map_decorator
def naive_map(steps, times, values):
    map_enter = time.perf_counter()
    time.sleep(0.001)  # 시간 소비 스텝
    time.sleep(0.0001)  # 메모리 소비 스텝
    map_elapsed = time.perf_counter() - map_enter
    return (
        tf.concat((steps, [["Map"]]), axis=0),
        tf.concat((times, [[map_enter, map_elapsed]]), axis=0),
        tf.concat((values, [values[-1]]), axis=0)
    )

@map_decorator
def time_consumming_map(steps, times, values):
    map_enter = time.perf_counter()
    time.sleep(0.001 * values.shape[0])  # 시간 소비 스텝
    map_elapsed = time.perf_counter() - map_enter

    return (
        tf.concat((steps, tf.tile([[["1st map"]]], [steps.shape[0], 1, 1])), axis=1),
        tf.concat((times, tf.tile([[[map_enter, map_elapsed]]], [times.shape[0], 1, 1])), axis=1),
        tf.concat((values, tf.tile([[values[:][-1][0]]], [values.shape[0], 1, 1])), axis=1)
    )

@map_decorator
def memory_consumming_map(steps, times, values):
    map_enter = time.perf_counter()
    time.sleep(0.0001 * values.shape[0])  # 메모리 소비 스텝
    map_elapsed = time.perf_counter() - map_enter
    # 배치 차원을 다루는 데 tf.tile 사용
    return (
        tf.concat((steps, tf.tile([[["2nd map"]]], [steps.shape[0], 1, 1])), axis=1),
        tf.concat((times, tf.tile([[[map_enter, map_elapsed]]], [times.shape[0], 1, 1])), axis=1),
        tf.concat((values, tf.tile([[values[:][-1][0]]], [values.shape[0], 1, 1])), axis=1)
    )


def timelined_benchmark(dataset, num_epochs=2):
    # 누산기 초기화
    steps_acc = tf.zeros([0, 1], dtype=tf.dtypes.string)
    times_acc = tf.zeros([0, 2], dtype=tf.dtypes.float32)
    values_acc = tf.zeros([0, 3], dtype=tf.dtypes.int32)

    start_time = time.perf_counter()
    for epoch_num in range(num_epochs):
        epoch_enter = time.perf_counter()
        for (steps, times, values) in dataset:
            # 데이터셋 준비 정보 기록하기
            steps_acc = tf.concat((steps_acc, steps), axis=0)
            times_acc = tf.concat((times_acc, times), axis=0)
            values_acc = tf.concat((values_acc, values), axis=0)

            # 훈련 시간 시뮬레이션
            train_enter = time.perf_counter()
            time.sleep(0.01)
            train_elapsed = time.perf_counter() - train_enter

            # 훈련 정보 기록하기
            steps_acc = tf.concat((steps_acc, [["Train"]]), axis=0)
            times_acc = tf.concat((times_acc, [(train_enter, train_elapsed)]), axis=0)
            values_acc = tf.concat((values_acc, [values[-1]]), axis=0)

        epoch_elapsed = time.perf_counter() - epoch_enter
        # 에포크 정보 기록하기
        steps_acc = tf.concat((steps_acc, [["Epoch"]]), axis=0)
        times_acc = tf.concat((times_acc, [(epoch_enter, epoch_elapsed)]), axis=0)
        values_acc = tf.concat((values_acc, [[-1, epoch_num, -1]]), axis=0)
        time.sleep(0.001)

    tf.print("실행 시간:", time.perf_counter() - start_time)
    return {"steps": steps_acc, "times": times_acc, "values": values_acc}        


naive_timeline = timelined_benchmark(
    tf.data.Dataset.range(2)
    .flat_map(dataset_generator_fun)
    .map(naive_map)
    .batch(_batch_map_num_items, drop_remainder=True)
    .unbatch(),
    5
)

optimized_timeline = timelined_benchmark(
    tf.data.Dataset.range(2)
    .interleave(  # 데이터 읽기 병렬화
        dataset_generator_fun,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(  # 매핑된 함수 벡터화
        _batch_map_num_items,
        drop_remainder=True)
    .map(  # 맵 변환 병렬화
        time_consumming_map,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .cache()  # 데이터 캐시
    .map(  # 메모리 사용량 줄이기
        memory_consumming_map,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .prefetch(  # 프로듀서와 컨슈머 작업 오버랩
        tf.data.experimental.AUTOTUNE
    )
    .unbatch(),
    5
)

draw_timeline(naive_timeline, title="Naive", width=10, save=False)
draw_timeline(optimized_timeline, title="Optimized", width=10, save=False)
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
#### Pipeline Optimization&Evaluation
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
#### Pipeline Optimization&Evaluation
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
#### Pipeline Optimization&Evaluation
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
#### Pipeline Optimization&Evaluation
```python
```


---




