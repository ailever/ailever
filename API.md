# Ailever API
## ailever.apps
```python
from ailever import apps
apps.run()
```


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
from ailever.forecasting import TSA

tsa = TSA()
tsa.analyze(time_series=time_series)
tsa.predict(predict_range=1.7)
```

## ailever.utils
`Debugging`
```python
from ailever.utils import Debug
debug = Debug()
debug(object, logname='logname')
del debug

from ailever.utils import Torchbug
torchbug = Torchbug()
torchbug(tensor, logname='logname')
del torchbug
```

`Download`
```python
from ailever.utils import storage
stroage('list')
storage('project.tar.gz')

from ailever.utils import repository
repository('list')
repository('ailever')
repository('ailever', tree=True)
repository('ailever', path='READMD.md')
repository('programming-language')
repository('programming-language', tree=True)
repository('programming-language', path='READMD.md')
repository('numerical-method')
repository('numerical-method', tree=True)
repository('numerical-method', path='READMD.md')
repository('statistics')
repository('statistics', tree=True)
repository('statistics', path='READMD.md')
repository('deep-learning')
repository('deep-learning', tree=True)
repository('deep-learning', path='READMD.md')
repository('reinforcement-learning')
repository('reinforcement-learning', tree=True)
repository('reinforcement-learning', path='READMD.md')
repository('applications')
repository('applications', tree=True)
repository('applications', path='READMD.md')

from ailever.utils import cloud
cloud('list')
cloud('materials.pdf')
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
[Numpy](https://github.com/ailever/ailever/blob/master/API.md)|
[Scipy](https://docs.scipy.org/doc/scipy/reference/)|
[Sympy](https://docs.sympy.org/latest/py-modindex.html)|
[Statsmodels](https://www.statsmodels.org/devel/api.html)|
[arch](https://arch.readthedocs.io/en/latest/api.html)|
[Scikit-learn](https://scikit-learn.org/stable/modules/classes.html#)|
[Matplotlib](https://matplotlib.org/api/index.html)|
[Seaborn](https://seaborn.pydata.org/api.html#)|
[Pandas](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)|
[Pyotrch](https://pytorch.org/docs/stable/index.html)|
[Transformers](https://huggingface.co/transformers/index.html)|
[R](https://cran.r-project.org/manuals.html)|

- https://github.com/ailever/ailever/tree/master/libraries
- https://github.com/bharathgs/Awesome-pytorch-list
- https://d2l.ai/index.html

- [GAN(github)](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations)
- [parlai](https://parl.ai/)


