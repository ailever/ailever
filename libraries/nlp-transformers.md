## [Natural Language Processing] | [huggingface](https://huggingface.co/) | [github](https://github.com/huggingface/transformers)
- [organizations](https://huggingface.co/organizations)
- [transformers main](https://huggingface.co/transformers/index.html)
- [modules](https://huggingface.co/transformers/_modules/index.html)
- [nlp](https://huggingface.co/nlp/), [github](https://github.com/huggingface/nlp)
- [datasets](https://huggingface.co/datasets)
- [models](https://huggingface.co/models)
- [metric](https://huggingface.co/metrics)
- [visualization(exbert)](https://exbert.net/), [github](https://github.com/bhoov/exbert)


## English

## Korean
- https://github.com/SKTBrain/KoBERT
- https://github.com/monologg/KoBERT-Transformers

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
bert_model = BertModel.from_pretrained('monologg/kobert')
bert_model(**tokenizer('안녕, 나는 자연어처리를 담당해.', return_tensors='pt'))
bert_model(**tokenizer(['안녕, 나는 자연어처리를 담당해.', '버트모델을 사용해보자!'], return_tensors='pt', padding=True, truncation=True))
```

```python
from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('monologg/kobert')
distilbert_model = DistilBertModel.from_pretrained('monologg/distilkobert')
distilbert_model(**tokenizer('안녕, 나는 자연어처리를 담당해.', return_tensors='pt'))
distilbert_model(**tokenizer(['안녕, 나는 자연어처리를 담당해.', '버트모델을 사용해보자!'], return_tensors='pt', padding=True, truncation=True))
```
