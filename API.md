# Ailever API
## ailever.language
```python
from ailever.language import sentiment
sentiment(sentence='')

from ailever.language import answer
answer(question='', context='')

from ailever.language import summary
summary(article='')

from ailever.language import generation
generation(sentence='')

```

## ailever.captioning
```python
from ailever import captioning

```

## ailever.detection
```python
from ailever import detection

```

## ailever.forecasting
```python
from ailever import forecasting

```

## ailever.utils

```python
from ailever.utils import Debug
debug = Debug()
debug(object, logname='logname')
del debug

from ailever.utils import Torchbug
torchbug = Torchbug()
torchbug(tensor, logname='logname')
del torchbug


from ailever.utils import storage
storage('project.tar.gz')

from ailever.utils import repository
repository('ailever')
repository('programming-language')
repository('numerical-method')
repository('applications')
repository('deep-learning')
```

### ailever.utils.data

```python
from ailever.utils.data import generator
generator(num=1000, save=True, visualize=True)

from ailever.utils.data import visualizer
visualizer(dataset)
```

# AI research teams
- https://ai.facebook.com/
- https://github.com/facebookresearch
- https://github.com/openai
- https://github.com/google-research
- https://github.com/deepmind
- https://github.com/clovaai
- https://github.com/kakaobrain


# API of Other libraries
https://github.com/bharathgs/Awesome-pytorch-list

- [Pyotrch](https://pytorch.org/docs/stable/index.html), [github](https://github.com/pytorch/pytorch)
- [Transformers](https://huggingface.co/transformers/index.html), [github](https://github.com/huggingface/transformers)
  - [exbert](https://exbert.net/), [github](https://github.com/bhoov/exbert)
- [GAN(github)](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations)


[Numpy](https://github.com/ailever/ailever/blob/master/API.md)|
[Scipy](https://docs.scipy.org/doc/scipy/reference/)|
[Sympy](https://docs.sympy.org/latest/py-modindex.html)|
[Statsmodels](https://www.statsmodels.org/devel/api.html)|
[Scikit-learn](https://scikit-learn.org/stable/modules/classes.html#)|
[Matplotlib](https://matplotlib.org/api/index.html)|
[Seaborn](https://seaborn.pydata.org/api.html#)|
[Pandas](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)|
