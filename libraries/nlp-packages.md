## Packages
### sklearn
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "Peace is a concept of societal friendship and harmony in the absence of hostility and violence.",
    "In a social sense, peace is commonly used to mean a lack of conflict (such as war) and freedom from fear of violence between individuals or groups.", 
    " Throughout history, leaders have used peacemaking and diplomacy to establish a type of behavioral restraint that has resulted in the establishment of regional peace or economic growth through various forms of agreements or peace treaties.",
    "Such behavioral restraint has often resulted in the reduced conflict, greater economic interactivity, and consequently substantial prosperity."
]
vector = CountVectorizer()
vector.fit_transform(corpus).toarray()
vector.vocabulary_
```
```
array([[1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
       [0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1],
       [0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 3, 0, 2, 2, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
       [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

{'peace': 37,
 'is': 30,
 'concept': 7,
 'of': 34,
 'societal': 46,
 'friendship': 17,
 'and': 2,
...
 'consequently': 9,
 'substantial': 47,
 'prosperity': 39}
```



```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Peace is a concept of societal friendship and harmony in the absence of hostility and violence.",
    "In a social sense, peace is commonly used to mean a lack of conflict (such as war) and freedom from fear of violence between individuals or groups.", 
    " Throughout history, leaders have used peacemaking and diplomacy to establish a type of behavioral restraint that has resulted in the establishment of regional peace or economic growth through various forms of agreements or peace treaties.",
    "Such behavioral restraint has often resulted in the reduced conflict, greater economic interactivity, and consequently substantial prosperity."
]

tfidfv = TfidfVectorizer()
tfidfv.fit_transform(corpus).toarray()
tfidfv.vocabulary_
```
```
array([[0.30083876, 0.        , 0.31398029, 0.        , 0.        , 0.        , 0.        , 0.30083876, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.30083876, 0.        , 0.        , 0.        , 0.        , 0.30083876, 0.        , 0.        , 0.        , 0.30083876, 0.15699015, 0.        , 0.        , 0.23718474, 0.        , 0.        , 0.        , 0.38404297, 0.        , 0.        , 0.19202149, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.30083876, 0.        , 0.        , 0.        , 0.19202149, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.23718474, 0.        ],
       [0.        , 0.        , 0.11688372, 0.2239832 , 0.        , 0.2239832 , 0.2239832 , 0.        , 0.17659092, 0.        , 0.        , 0.        , 0.        , 0.        , 0.2239832 , 0.        , 0.2239832 , 0.        , 0.2239832 , 0.        , 0.2239832 , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.11688372, 0.2239832 , 0.        , 0.17659092, 0.2239832 , 0.        , 0.2239832 , 0.28593114, 0.        , 0.17659092, 0.14296557, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.2239832 , 0.2239832 , 0.        , 0.        , 0.17659092, 0.        , 0.        , 0.        , 0.        , 0.17659092, 0.        , 0.        , 0.17659092, 0.        , 0.17659092, 0.2239832 ],
       [0.        , 0.18231336, 0.09513867, 0.        , 0.14373794, 0.        , 0.        , 0.        , 0.        , 0.        , 0.18231336, 0.14373794, 0.18231336, 0.18231336, 0.        , 0.18231336, 0.        , 0.        , 0.        , 0.        , 0.        , 0.18231336, 0.        , 0.14373794, 0.18231336, 0.18231336, 0.        , 0.09513867, 0.        , 0.        , 0.        , 0.        , 0.18231336, 0.        , 0.34910476, 0.        , 0.28747589, 0.23273651, 0.18231336, 0.        , 0.        , 0.18231336, 0.14373794, 0.14373794, 0.        , 0.        , 0.        , 0.        , 0.        , 0.18231336, 0.11636825, 0.18231336, 0.18231336, 0.14373794, 0.18231336, 0.18231336, 0.14373794, 0.18231336, 0.        , 0.        ],
       [0.        , 0.        , 0.14877489, 0.        , 0.22477291, 0.        , 0.        , 0.        , 0.22477291, 0.28509594, 0.        , 0.22477291, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.28509594, 0.        , 0.        , 0.        , 0.22477291, 0.        , 0.        , 0.        , 0.14877489, 0.        , 0.28509594, 0.        , 0.        , 0.        , 0.        , 0.        , 0.28509594, 0.        , 0.        , 0.        , 0.28509594, 0.28509594, 0.        , 0.22477291, 0.22477291, 0.        , 0.        , 0.        , 0.28509594, 0.22477291, 0.        , 0.18197304, 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        , 0.        ]])

{'peace': 37,
 'is': 30,
 'concept': 7,
 'of': 34,
 'societal': 46,
 'friendship': 17,
 'and': 2,
...
 'consequently': 9,
 'substantial': 47,
 'prosperity': 39}
```

### tensorflow
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

### transformers
```python
```

### nlpy
```python
```

### sentencepiece
`Installation`
```bash
$ pip install sentencepiece
```


### tokenizers
`Installation`
```bash
$ pip install tokenizers
```


---

## Korean Language Preprocessing Packages
### KoNLPY
`Installation` : https://konlpy.org/ko/latest/install/#ubuntu
```bash
$ sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl
$ python3 -m pip install --upgrade pip
$ python3 -m pip install konlpy       # Python 3.x
$ sudo apt-get install curl git
$ bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```
```python
from konlpy.tag import Okt

sentence = "유구한 역사와 전통에 빛나는 우리 대한국민은 3·1운동으로 건립된 대한민국임시정부의 법통과 불의에 항거한 4·19민주이념을 계승하고, 조국의 민주개혁과 평화적 통일의 사명에 입각하여 정의·인도와 동포애로써 민족의 단결을 공고히 하고, 모든 사회적 폐습과 불의를 타파하며, 자율과 조화를 바탕으로 자유민주적 기본질서를 더욱 확고히 하여 정치·경제·사회·문화의 모든 영역에 있어서 각인의 기회를 균등히 하고, 능력을 최고도로 발휘하게 하며, 자유와 권리에 따르는 책임과 의무를 완수하게 하여, 안으로는 국민생활의 균등한 향상을 기하고 밖으로는 항구적인 세계평화와 인류공영에 이바지함으로써 우리들과 우리들의 자손의 안전과 자유와 행복을 영원히 확보할 것을 다짐하면서 1948년 7월 12일에 제정되고 8차에 걸쳐 개정된 헌법을 이제 국회의 의결을 거쳐 국민투표에 의하여 개정한다."

okt = Okt()
okt.morphs(sentence)
```
```
['유구',
 '한',
...
 '개정',
 '한다',
 '.']
```

### PyKoSpacing
`Installation` : https://github.com/haven-jeon/PyKoSpacing
```bash
$ pip install git+https://github.com/haven-jeon/PyKoSpacing.git
```
```python
from pykospacing import Spacing

sentence = "유구한 역사와 전통에 빛나는 우리 대한국민은 3·1운동으로 건립된 대한민국임시정부의 법통과 불의에 항거한 4·19민주이념을 계승하고, 조국의 민주개혁과 평화적 통일의 사명에 입각하여 정의·인도와 동포애로써 민족의 단결을 공고히 하고, 모든 사회적 폐습과 불의를 타파하며, 자율과 조화를 바탕으로 자유민주적 기본질서를 더욱 확고히 하여 정치·경제·사회·문화의 모든 영역에 있어서 각인의 기회를 균등히 하고, 능력을 최고도로 발휘하게 하며, 자유와 권리에 따르는 책임과 의무를 완수하게 하여, 안으로는 국민생활의 균등한 향상을 기하고 밖으로는 항구적인 세계평화와 인류공영에 이바지함으로써 우리들과 우리들의 자손의 안전과 자유와 행복을 영원히 확보할 것을 다짐하면서 1948년 7월 12일에 제정되고 8차에 걸쳐 개정된 헌법을 이제 국회의 의결을 거쳐 국민투표에 의하여 개정한다."

spacing = Spacing()
kospacing_sent = spacing(sentence.replace(' ', '')) 
kospacing_sent
```
```
'유구한 역사와 전통에 빛나는 우리 대한 국민은 3·1운동으로 건립된 대한민국 임시정부의 법 통과 불의에 항거한 4·19 민주이념을 계승하고, 조국의 민주개혁과 평화적 통일의 사명에 입각하여 정의·인도와 동포 애로써 민족의 단결을 공고히 하고, 모든 사회적 폐습과 불의를 타파하며, 자율과 조화를 바탕으로 자유민주적 기본질서를 더욱 확고히 하여 정치·경제·사회·문화의 모든 영역에 있어서 각인의 기회를 균등히 하고, 능력을 최고 도로 발휘하게 하며, 자유와권 리에 따르는 책임과 의무를 완수하게 하여, 안으로는 국민생활의 균등한 향상을 기하고 밖으로는 항구적인 세계평화와 인류공영에 이바지함으로써 우리들과 우리들의 자손의 안전과 자유와 행복을 영원히 확보할 것을 다짐하면서 1948년 7월 12일에 제정되고 8차에 걸쳐 개정된 헌법을 이제 국회의 의결을 거쳐 국민투표에 의하여 개정한다.'
```

### Py-Hanspell
`Installation` : https://github.com/ssut/py-hanspell
```bash
$ pip install git+https://github.com/ssut/py-hanspell.git
```
```python
from hanspell import spell_checker

sentence = "유구한 역사와 전통에 빛나는 우리 대한국민은 3·1운동으로 건립된 대한민국임시정부의 법통과 불의에 항거한 4·19민주이념을 계승하고, 조국의 민주개혁과 평화적 통일의 사명에 입각하여 정의·인도와 동포애로써 민족의 단결을 공고히 하고, 모든 사회적 폐습과 불의를 타파하며, 자율과 조화를 바탕으로 자유민주적 기본질서를 더욱 확고히 하여 정치·경제·사회·문화의 모든 영역에 있어서 각인의 기회를 균등히 하고, 능력을 최고도로 발휘하게 하며, 자유와 권리에 따르는 책임과 의무를 완수하게 하여, 안으로는 국민생활의 균등한 향상을 기하고 밖으로는 항구적인 세계평화와 인류공영에 이바지함으로써 우리들과 우리들의 자손의 안전과 자유와 행복을 영원히 확보할 것을 다짐하면서 1948년 7월 12일에 제정되고 8차에 걸쳐 개정된 헌법을 이제 국회의 의결을 거쳐 국민투표에 의하여 개정한다."

spelled_sent = spell_checker.check(sentence.replace(' ', ''))
hanspell_sent = spelled_sent.checked
hanspell_sent
```
```
'유구한 역사와 전통에 빛나는 우리 대한 국민은 3·1운동으로 건립된 대한민국임시정부의 법통과 불의에 항거한 4·19민주이념을 계승하고, 조국의 민주개혁과 평화적 통일의 사명에 입각하여 정의·인도와 동포애로써 민족의 단결을 공고히 하고, 모든 사회적 폐습과 불의를 타파하며, 자율과 조화를 바탕으로 자유민주적 기본질서를 더욱 확고히 하여 정치·경제·사회·문화의 모든 영역에 있어서 각인의 기회를 균등히 하고, 능력을 최고도로 발휘하게 하며, 자유와 권리에 따르는 책임과 의무를 완수하게 하여, 안으로는 국민 생활의 균등한 향상을 기하고 밖으로는 항구적인 세계 평화와 인류공영에 이바지함으로써 우리들과 우리들의 자손의 안전과 자유와 행복을 영원히 확보할 것을 다짐하면서 1948년 7월 12일에 제정되고 8차에 걸쳐 개정된 헌법을 이제 국회의 의결을 거쳐 국민투표에 의하여 개정한다.'
```

### SOYNLP
`Installation` : https://github.com/lovit/soynlp
```bash
$ pip install soynlp
```


### Customized KoNLPy
`Installation` : https://github.com/lovit/customized_konlpy
```bash
$ pip install customized_konlpy
```
```python
from ckonlpy.tag import Twitter

sentence = "유구한 역사와 전통에 빛나는 우리 대한국민은 3·1운동으로 건립된 대한민국임시정부의 법통과 불의에 항거한 4·19민주이념을 계승하고, 조국의 민주개혁과 평화적 통일의 사명에 입각하여 정의·인도와 동포애로써 민족의 단결을 공고히 하고, 모든 사회적 폐습과 불의를 타파하며, 자율과 조화를 바탕으로 자유민주적 기본질서를 더욱 확고히 하여 정치·경제·사회·문화의 모든 영역에 있어서 각인의 기회를 균등히 하고, 능력을 최고도로 발휘하게 하며, 자유와 권리에 따르는 책임과 의무를 완수하게 하여, 안으로는 국민생활의 균등한 향상을 기하고 밖으로는 항구적인 세계평화와 인류공영에 이바지함으로써 우리들과 우리들의 자손의 안전과 자유와 행복을 영원히 확보할 것을 다짐하면서 1948년 7월 12일에 제정되고 8차에 걸쳐 개정된 헌법을 이제 국회의 의결을 거쳐 국민투표에 의하여 개정한다."

twitter = Twitter()
twitter.add_dictionary('대한국민', 'Noun')
twitter.morphs(sentence)
```
```
['유구',
 '한',
...
 '개정',
 '한',
 '다',
 '.']
```
