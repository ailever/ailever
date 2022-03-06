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

### tensorflow
```python
```

### transformers
```python
```

### nlpy
```python
```



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
