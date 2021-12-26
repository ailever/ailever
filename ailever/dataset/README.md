# Dataset Package
https://github.com/ailever/dataset

### Training Dataset by learning-problem type
```python
from ailever.dataset import UCI, SMAPI, SKAPI

# regression(1)
dataset = SMAPI.macrodata(download=False).rename(columns={'infl':'target'})
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, 'target'].ravel()

# regression(2)
dataset = UCI.beijing_airquality(download=False).rename(column={'pm2.5':'target'})
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, 'target'].ravel()

# classification(1)
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, 'target'].ravel()

# classification(2)
dataset = SKAPI.digits(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, 'target'].ravel()

# classification(3)
dataset = UCI.adult(download=False).rename(column={'50K':'target'})
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, 'target'].ravel()
```

### From Ailever Repository
```python
from ailever.dataset import Loader

Loader.______(download=False)
```


### UCI Machine Learning Repository
```python
from ailever.dataset import UCI

UCI.adult(download=True)
UCI.abalone(download=True)
UCI.maintenance(download=True)
UCI.beijing_airquality(download=True)
UCI.white_wine(download=True)
UCI.red_wine(download=True)
UCI.annealing(download=True)
UCI.breast_cancer(download=True)
```

### From Statsmodels API
```python
from ailever.dataset import SMAPI

SMAPI.macrodata(download=True)
SMAPI.sunspots(download=True)
SMAPI.anes96(download=True)
SMAPI.cancer(download=True)
SMAPI.ccard(download=True)
SMAPI.china_smoking(download=True)
SMAPI.co2(download=True)
SMAPI.committee(download=True)
SMAPI.copper(download=True)
SMAPI.cpunish(download=True)
SMAPI.elnino(download=True)
SMAPI.engel(download=True)
SMAPI.fair(download=True)
SMAPI.fertility(download=True)
SMAPI.grunfeld(download=True)
SMAPI.heart(download=True)
SMAPI.interest_inflation(download=True)
SMAPI.longley(download=True)
SMAPI.modechoice(download=True)
SMAPI.nile(download=True)
SMAPI.randhie(download=True)
SMAPI.scotland(download=True)
SMAPI.spector(download=True)
SMAPI.stackloss(download=True)
SMAPI.star98(download=True)
SMAPI.statecrime(download=True)
SMAPI.strikes(download=True)
```

### From Scikit-Learn API
```python
from ailever.dataset import SKAPI

SKAPI.housing(download=True)
SKAPI.breast_cancer(download=True)
SKAPI.diabetes(download=True)
SKAPI.digits(download=True)
SKAPI.iris(download=True)
SKAPI.wine(download=True)
```

### From Ailever API
```python
from ailever.dataset import AILAPI

AILAPI(meta_info=True)
AILAPI(table='district0001', download=True)
```
